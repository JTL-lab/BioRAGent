import os
import re
import oci
import json
import pymupdf
import numpy as np
from tqdm import tqdm
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, explode, lit
from pyspark.sql.types import StringType, StructType, StructField, ArrayType
from langchain.text_splitter import RecursiveCharacterTextSplitter
from oci.generative_ai_inference import GenerativeAiInferenceClient


def extract_text_from_pdf(pdf_path):
    """
    Extracts PDF file text (disregarding figures/tables).
    """
    try:
        doc = pymupdf.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return ""


@udf(returnType=StringType())
def extract_text_udf(pdf_path):
    return extract_text_from_pdf(pdf_path)


@udf(returnType=StringType())
def clean_pdf_text_udf(text):
    if not text:
        return ""
    
    text = re.sub(r'\|\d+\.\d+v\d+\.pdf\|', '', text)                                      # PDF file markers (e.g., |2412.19191v1.pdf|)
    text = re.sub(r'arXiv:\d+\.\d+v\d+\s+\[\w+-?\w*\.\w+\]\s+\d+\s+\w+\s+\d{4}', '', text) # arXiv identifiers
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)                                         # hyphenated words at line breaks
    text = re.sub(r'\n\n+', '[PARA]', text)                                                # replace multiple newlines with paragraph markers
    text = re.sub(r'\n', ' ', text)                                                        # replace single newlines with spaces
    text = re.sub(r'\n(?=[a-z])', ' ', text)                                               # get rid of newlines that occur mid-sentence just signifying a natural newline in formatting,
    text = re.sub(r'([^.!?])\n([^A-Z])', r'\1 \2', text)                                   # but still keep structural newlines (e.g. after a period, followed by uppercase suggesting new paragraph, etc.)
    text = text.replace('[PARA]', '\n\n')                                                  # restore paragraph breaks
    text = re.sub(r'\s+', ' ', text)                                                       # fix spacing issues
    text = re.sub(r'\s*\.\s*', '. ', text)
    text = re.sub(r'\s*,\s*', ', ', text)

    # arXiv articles can have some other weird symbols after being preprocessed with pymupdf so need to handle these too
    text = re.sub(r'•', '- ', text)  # bullet points
    text = re.sub(r'…', '...', text)  # ellipses
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)  # get rid of any control chars
    
    return text.strip()


@udf(returnType=StringType())
def extract_metadata_udf(file_name, text):
    metadata = {"file_name": file_name} # Filename 
    # See if we can find the title within the text
    title_match = re.search(r'(?:TITLE:|^)([^\n]+)', text, re.IGNORECASE)
    if title_match:
        metadata["title"] = title_match.group(1).strip()
    
    # Try to find abstract
    abstract_match = re.search(r'(?:ABSTRACT|Summary)(?:\s*\n+\s*)(.*?)(?:\n\n|\n\d\.|\nINTRODUCTION)', 
                                text, re.IGNORECASE | re.DOTALL)
    if abstract_match:
        metadata["abstract"] = abstract_match.group(1).strip()
    
    # also store arXiv ID if present
    arxiv_match = re.search(r'arXiv:(\d+\.\d+v\d+)', text)
    if arxiv_match:
        metadata["arxiv_id"] = arxiv_match.group(1)
    
    return json.dumps(metadata)


@udf(returnType=ArrayType(StructType([
        StructField("chunk_text", StringType(), True),
        StructField("chunk_metadata", StringType(), True)
    ])))
def semantic_chunk_udf(text, metadata_json, chunk_size=2000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]  # try to split on paragraphs first, but end of sentences if not, and worst case after word
    )
    
    if not text or len(text.strip()) == 0:
        return []
    
    try:
        metadata = json.loads(metadata_json)
    except:
        metadata = {}
        
    sections = []
    section_matches = re.finditer(r'(?:\n|\A)(\d+(?:\.\d+)?[\s\t]+[A-Z][A-Z\s]+)(?:\n|\Z)', text) # check for section headers (e.g., "1. INTRODUCTION", "2.1 METHODS")
    last_pos = 0
    section_positions = []
    
    for match in section_matches:
        section_positions.append((last_pos, match.start(), None))
        last_pos = match.start()
        section_positions.append((match.start(), match.end(), match.group(1).strip()))
    section_positions.append((last_pos, len(text), None))
    
    # ideally use sections when possible for chunking to maintain semantic units
    chunks_with_metadata = []
    if len(section_positions) > 1:
        for start, end, section_title in section_positions:
            section_text = text[start:end].strip()
            if not section_text:
                continue
                
            if len(section_text) > chunk_size:
                section_chunks = text_splitter.split_text(section_text)
                for i, chunk in enumerate(section_chunks):
                    chunk_meta = metadata.copy()
                    if section_title:
                        chunk_meta["section"] = section_title
                    chunk_meta["chunk_index"] = i
                    chunks_with_metadata.append({"chunk_text": chunk, 
                                                "chunk_metadata": json.dumps(chunk_meta)})
            else:
                chunk_meta = metadata.copy()
                if section_title:
                    chunk_meta["section"] = section_title
                chunks_with_metadata.append({"chunk_text": section_text, 
                                            "chunk_metadata": json.dumps(chunk_meta)})
    else:
        # If no sections found, fall back to standard chunking
        regular_chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(regular_chunks):
            chunk_meta = metadata.copy()
            chunk_meta["chunk_index"] = i
            chunks_with_metadata.append({"chunk_text": chunk, 
                                        "chunk_metadata": json.dumps(chunk_meta)})
            
    return chunks_with_metadata


def run_spark_pdf_preprocessing(pdf_dir, 
                                file_list=None,
                                max_files=10, 
                                output_path="test_processed_papers.parquet", 
                                chunk_size=2000, chunk_overlap=100):
    """
    Process PDF files using PySpark and extract text
    
    Args:
        pdf_dir (str): Path to directory containing PDF files.
        file_list (list, optional): Specific list of files to process. If None, uses max_files.
        max_files (int): Maximum number of files to process. 
        output_path (str): Path to output directory to save resulting preprocessed data parquet file.
        chunk_size (int): Size of each chunk in characters.
        chunk_overlap (int): Overlap between chunks in characters.
    
    Returns:
        final_df (pd.DataFrame): DataFrame of chunked PDF text content.
    """
    spark = SparkSession.builder \
        .appName("BioRAGent-TextProcessing") \
        .config("spark.driver.memory", "50g") \
        .config("spark.executor.memory", "50g") \
        .config("spark.memory.offHeap.enabled", "true") \
        .config("spark.memory.offHeap.size", "16g") \
        .getOrCreate()
    
    schema = StructType([
        StructField("file_path", StringType(), True),
        StructField("file_name", StringType(), True),
        StructField("text", StringType(), True),
        StructField("metadata", StringType(), True)
    ])
    
    if file_list:
        pdf_files = [(os.path.join(pdf_dir, f), f, "", "") for f in file_list]
    else:
        pdf_files = [(os.path.join(pdf_dir, f), f, "", "") 
                    for f in os.listdir(pdf_dir)[:max_files] if f.endswith(".pdf")]
                    
    print(f"Processing {len(pdf_files)} PDF files")

    pdf_df = spark.createDataFrame(pdf_files, schema)
    pdf_df = pdf_df.repartition(min(200, max(20, len(pdf_files) // 10)))  # Dynamic partitioning
    
    # 1) Extract all text using extraction UDF
    pdf_df = pdf_df.withColumn("text", extract_text_udf(col("file_path")))
    pdf_df = pdf_df.filter(col("text").isNotNull() & (col("text") != ""))
    pdf_df.cache()
    
    # 2) Clean the extracted text
    pdf_df = pdf_df.withColumn("clean_text", clean_pdf_text_udf(col("text")))

    # 3) Retain metadata from text
    pdf_df = pdf_df.withColumn("metadata", extract_metadata_udf(col("file_name"), col("clean_text")))
    pdf_df.checkpoint()
    pdf_df.count()
    
    # 4) Semantic chunking prior to processing with RAG agent (need to consider max chunk sizes)
    pdf_df = pdf_df.withColumn("semantic_chunks", 
                              semantic_chunk_udf(col("clean_text"), col("metadata"), 
                                              lit(chunk_size), lit(chunk_overlap)))
    
    final_df = pdf_df.select(
        col("file_path"),
        col("file_name"),
        explode(col("semantic_chunks")).alias("chunk_data")
    ).select(
        col("file_path"),
        col("file_name"),
        col("chunk_data.chunk_text"),
        col("chunk_data.chunk_metadata")
    )
    
    total_chunks = final_df.count()
    total_docs = pdf_df.count()
    print(f"Total documents processed: {total_docs}")
    print(f"Total chunks generated: {total_chunks}")
    print(f"Average chunks per document: {total_chunks/total_docs:.2f}")
    
    if output_path:
        final_df.write.mode("overwrite").parquet(output_path)
        print(f"Saved chunked results to {output_path}")
    
    return final_df


def run_spark_pdf_batch_processing(pdf_dir, batch_size=1000, output_path="processed_papers.parquet", 
                                  chunk_size=2000, chunk_overlap=100):
    """
    Processes PDF files in batches and writes data to a single output Parquet file.
    
    Args:
        pdf_dir (str): Path to directory containing PDF files.
        batch_size (int): Number of files to process in each batch.
        output_path (str): Path to output Parquet file.
        chunk_size (int): Size of each chunk in characters.
        chunk_overlap (int): Overlap between chunks in characters.
    """
    spark = SparkSession.builder \
        .appName("BioRAGent-BatchProcessing") \
        .config("spark.driver.memory", "50g") \
        .config("spark.executor.memory", "50g") \
        .config("spark.memory.offHeap.enabled", "true") \
        .config("spark.memory.offHeap.size", "16g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.default.parallelism", "100") \
        .getOrCreate()
    spark.sparkContext.setCheckpointDir("/tmp/spark_checkpoint")

    all_pdf_files = sorted([f for f in os.listdir(pdf_dir) if f.endswith(".pdf")])
    total_files = len(all_pdf_files)
    total_batches = (total_files + batch_size - 1) // batch_size
    print(f"Found {total_files} PDF files to process in {total_batches} batches")
    
    # Create progress tracking file
    progress_file = os.path.join(os.path.dirname(output_path), "batch_progress.txt")
    last_completed_batch = -1
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            try:
                last_completed_batch = int(f.read().strip())
                print(f"Resuming from batch {last_completed_batch + 1}")
            except:
                print("Could not read previous progress, starting from batch 0")
    
    for batch_num in range(last_completed_batch + 1, total_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, total_files)
        batch_files = all_pdf_files[start_idx:end_idx]
        
        print(f"Processing batch {batch_num+1}/{total_batches} with {len(batch_files)} files")
        
        try:
            # Modify run_spark_pdf_preprocessing to accept a file_list
            final_df = run_spark_pdf_preprocessing(
                pdf_dir=pdf_dir, 
                file_list=batch_files,
                output_path=None, 
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            if batch_num == 0: 
                write_mode = "overwrite"
            else: 
                write_mode = "append"
            final_df.write.mode(write_mode).parquet(output_path)
            
            with open(progress_file, 'w') as f:
                f.write(str(batch_num))
            spark.catalog.clearCache()
            
            print(f"Completed batch {batch_num+1}/{total_batches}")
            
        except Exception as e:
            print(f"Error processing batch {batch_num}: {e}")
    
    print(f"Batch processing complete. Results written to {output_path}")


if __name__ == '__main__':
    """
    Main function to orchestrate the entire workflow
    """
    print("Preprocessing PDFs with PySpark...")
    pdf_dir = ""
    output_path = "all_arxiv_papers_processed.parquet"
    
    run_spark_pdf_batch_processing(
        pdf_dir=pdf_dir,
        batch_size=1000,
        output_path=output_path,
        chunk_size=2000,
        chunk_overlap=100
    )
