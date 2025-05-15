import os
import oci
import json
import pymupdf
import numpy as np
from tqdm import tqdm
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, StructType, StructField
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


def run_spark_pdf_preprocessing(pdf_dir, max_files=10, output_path="test_processed_papers.parquet"):
    """
    Process PDF files using PySpark and extract text
    
    Args:
        pdf_dir (str): Path to directory containing PDF files.
        max_files (int): Maximum number of files to process. 
        output_path (str): Path to output directory to save resulting preprocessed data parquet file.
    
    Returns:
        pdf_df (pd.DataFrame): DataFrame of PDF text content, with one PDF per row. 
    """
    spark = SparkSession.builder \
        .appName("BioRAGent-TextProcessing-Test") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .getOrCreate()
    
    schema = StructType([
        StructField("file_path", StringType(), True),
        StructField("file_name", StringType(), True),
        StructField("text", StringType(), True),
        StructField("metadata", StringType(), True)
    ])
    
    pdf_files = [(os.path.join(pdf_dir, f), f, "", "") 
                for f in os.listdir(pdf_dir)[:max_files] if f.endswith(".pdf")]
    print(f"Processing {len(pdf_files)} PDF files for testing")

    pdf_df = spark.createDataFrame(pdf_files, schema)
    pdf_df = pdf_df.withColumn("text", extract_text_udf(col("file_path")))

    total_files = pdf_df.count()
    successful_extractions = pdf_df.filter(col("text").isNotNull() & (col("text") != "")).count()    
    print(f"Total files processed: {total_files}")
    print(f"Successful extractions: {successful_extractions}")
    print(f"Failed extractions: {total_files - successful_extractions}")
    
    if output_path:
        pdf_df.write.mode("overwrite").parquet(output_path)
        print(f"Saved test results to {output_path}")
    
    return pdf_df


if __name__ == '__main__':
    """
    Main function to orchestrate the entire workflow
    """
    # Step 1: Preprocess PDFs using PySpark
    #print("Step 1: Preprocessing PDFs with PySpark...")
    #pdf_df = run_spark_pdf_preprocessing()
    #print(f"We have successfully processed {pdf_df.count()} PDF documents.")

