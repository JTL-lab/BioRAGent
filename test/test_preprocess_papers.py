import os
import json
import pymupdf
from tqdm import tqdm
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, StructType, StructField
from scripts.preprocess_papers import extract_text_from_pdf, run_spark_pdf_preprocessing

def test_preprocessing():
    """
    Test the PDF preprocessing function with a small subset.
    """
    df = run_spark_pdf_preprocessing(pdf_dir='', max_files=5)
    df.select("file_name", "text").show(5, truncate=False)
    df.select("file_name", col("text").substr(1, 100).alias("text_sample")).show(5)
    sample_file = df.select("file_path").first()[0]
    manual_text = extract_text_from_pdf(sample_file)

    print(f"\nManual extraction from {os.path.basename(sample_file)}:")
    print(f"Text length: {len(manual_text)}")
    print(f"Sample text: {manual_text[:200]}...")
    
    return df


def inspect_test_output_parquet(): 
    spark = SparkSession.builder.appName('InspectParquet').getOrCreate()
    df = spark.read.parquet('test_processed_papers.parquet')

    print('Schema:')
    df.printSchema()
    print('Sample data:')
    df.show(5, truncate=100)
    columns = df.columns
    print(f'Columns: {columns}')


if __name__ == "__main__":

    test_df = test_preprocessing()
    inspect_test_output_parquet()
