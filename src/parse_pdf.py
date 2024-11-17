from typing import Dict, List, Any, Optional
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import NarrativeText, Image as UnstructuredImage, Table, Formula
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pathlib import Path
from PIL import Image as PILImage
import base64
import matplotlib.pyplot as plt
import math
import os
import json
import re

class ExtractMetadata:
    """A class to extract and process metadata from PDF documents with LLM summaries."""

    def __init__(self, openai_api_key: Optional[str] = None, output_folder: str = "./data"):
        """
        Initialize the ExtractMetadata class.

        Args:
            openai_api_key (str): OpenAI API key for LLM services
            output_folder (str): Path to store extracted data
        """
        self.output_folder = Path(output_folder)
        self.image_dir = self.output_folder / "images"
        self.json_dir = self.output_folder / "pdf_extracts"
        
        # Create directories
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Get OpenAI API key
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided either through initialization or OPENAI_API_KEY environment variable")

        # Initialize LLM models
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            api_key=self.api_key
        )
        
        self.vision_model = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=1000,
            api_key=self.api_key
        )
        
        # Initialize prompts
        self.formula_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that explains mathematical formulas in detail. Focus on the meaning, applications, and implications of the formula."),
            ("user", "Explain the following mathematical formula in detail. Include its purpose, components, and practical applications:\n{formula_content}")
        ])
        
        self.image_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that provides detailed analysis of scientific and technical images."),
            ("user", "Provide a comprehensive and in depth summary of the image on page {page_number} of the document '{source_document}'. Focus on technical details, diagrams, charts, or any mathematical/scientific content shown.\n\n{image_content}")
        ])

        self.table_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that analyzes and summarizes tabular data from academic and technical documents."),
            ("user", "Provide a comprehensive summary of the following table from page {page_number}. Include key insights, patterns, and relationships in the data:\n\n{table_content}")
        ])
        
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a PDF and extract all components.

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            Dict[str, Any]: Extracted metadata and content
        """
        try:
            pdf_path = Path(pdf_path).resolve()
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            pdf_data = partition_pdf(
                filename=str(pdf_path),
                strategy="hi_res",
                extract_images_in_pdf=True,
                extract_image_block_to_payload=False,
                extract_image_block_output_dir=str(self.image_dir)
            )
            
            source_document = pdf_path.name
            
            results = {
                "metadata": {
                    "filename": source_document,
                    "filepath": str(pdf_path)
                },
                "content": {
                    "text": self.extract_text_with_metadata(pdf_data, source_document),
                    "images": self.extract_image_metadata(pdf_data, source_document),
                    "tables": self.extract_table_metadata(pdf_data, source_document),
                    "formulas": self.extract_formula_metadata(pdf_data, source_document)
                }
            }
            
            self.save_to_json(results, source_document)
            return results
            
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            raise

    def save_to_json(self, results: Dict[str, Any], source_document: str) -> None:
        """
        Save extracted content to JSON file.

        Args:
            results (Dict[str, Any]): Extracted data
            source_document (str): Source document name
        """
        safe_filename = re.sub(r'[^\w\-.]', '_', source_document)
        json_filename = f"{Path(safe_filename).stem}_extract.json"
        json_path = self.json_dir / json_filename
        
        try:
            json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f"Successfully saved extracts to {json_path}")
        except Exception as e:
            print(f"Error saving JSON file: {str(e)}")
            raise

    def extract_text_with_metadata(self, pdf_data: List[Any], source_document: str) -> List[Dict[str, Any]]:
        """
        Extract text content with metadata.

        Args:
            pdf_data (List[Any]): Parsed PDF data
            source_document (str): Source document name

        Returns:
            List[Dict[str, Any]]: List of text elements with metadata
        """
        text_data = []
        paragraph_counters: Dict[int, int] = {}

        for element in pdf_data:
            if isinstance(element, NarrativeText):
                page_number = getattr(element.metadata, 'page_number', 0)
                paragraph_counters[page_number] = paragraph_counters.get(page_number, 0) + 1

                text_data.append({
                    "source_document": source_document,
                    "page_number": page_number,
                    "paragraph_number": paragraph_counters[page_number],
                    "text": element.text.strip()
                })

        return text_data

    def extract_image_metadata(self, pdf_data: List[Any], source_document: str) -> List[Dict[str, Any]]:
        """
        Extract image metadata and generate summaries.

        Args:
            pdf_data (List[Any]): Parsed PDF data
            source_document (str): Source document name

        Returns:
            List[Dict[str, Any]]: List of image metadata with summaries
        """
        image_data = []

        for element in pdf_data:
            if isinstance(element, UnstructuredImage):
                page_number = getattr(element.metadata, 'page_number', 0)
                image_path = element.metadata.image_path if hasattr(element.metadata, 'image_path') else None

                if image_path:
                    try:
                        full_image_path = self.image_dir / Path(image_path).name
                        
                        # Read and encode image
                        with open(full_image_path, "rb") as image_file:
                            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                        
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": f"For page {page_number} of {source_document}"},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{encoded_string}",
                                            "detail": "auto"
                                        }
                                    }
                                ]
                            }
                        ]
                        summary = self.vision_model.invoke(messages).content
                        
                        image_data.append({
                            "source_document": source_document,
                            "page_number": page_number,
                            "image_path": str(full_image_path),
                            "summary": summary
                        })
                    except Exception as e:
                        print(f"Error processing image on page {page_number}: {str(e)}")
                        image_data.append({
                            "source_document": source_document,
                            "page_number": page_number,
                            "image_path": str(full_image_path),
                            "error": str(e)
                        })

        return image_data

    def extract_table_metadata(self, pdf_data: List[Any], source_document: str) -> List[Dict[str, Any]]:
        """
        Extract table content with metadata and generate summaries.

        Args:
            pdf_data (List[Any]): Parsed PDF data
            source_document (str): Source document name

        Returns:
            List[Dict[str, Any]]: List of table content with metadata and summaries
        """
        table_data = []

        for element in pdf_data:
            if isinstance(element, Table):
                page_number = getattr(element.metadata, 'page_number', 0)
                
                try:
                    table_content = str(element)
                    
                    # Generate summary using the OpenAI model
                    messages = self.table_prompt.format_messages(
                        table_content=table_content,
                        page_number=page_number
                    )
                    summary = self.llm.invoke(messages).content
                    
                    table_data.append({
                        "source_document": source_document,
                        "page_number": page_number,
                        "table_content": table_content,
                        "summary": summary
                    })
                except Exception as e:
                    print(f"Error processing table on page {page_number}: {str(e)}")

        return table_data

    def extract_formula_metadata(self, pdf_data: List[Any], source_document: str) -> List[Dict[str, Any]]:
        """
        Extract and process mathematical formulas.

        Args:
            pdf_data (List[Any]): Parsed PDF data
            source_document (str): Source document name

        Returns:
            List[Dict[str, Any]]: List of formulas with metadata
        """
        formula_data = []

        for element in pdf_data:
            if isinstance(element, Formula):
                page_number = getattr(element.metadata, 'page_number', 0)
                
                try:
                    formula = str(element)
                    formula_data.append({
                        "source_document": source_document,
                        "page_number": page_number,
                        "formula": formula.strip()
                    })
                except Exception as e:
                    print(f"Error processing formula on page {page_number}: {str(e)}")

        return formula_data

    def display_images(self, image_data: List[Dict[str, Any]], images_per_row: int = 4) -> None:
        """
        Display extracted images in a grid.

        Args:
            image_data (List[Dict[str, Any]]): List of image metadata
            images_per_row (int): Number of images to display per row
        """
        valid_images = [img for img in image_data if img.get('image_path')]
        if not valid_images:
            print("No valid image data available.")
            return

        num_images = len(valid_images)
        num_rows = math.ceil(num_images / images_per_row)

        fig, axes = plt.subplots(num_rows, images_per_row, figsize=(20, 5*num_rows))
        axes = axes.flatten() if num_rows > 1 else [axes]

        for idx, (ax, img_data) in enumerate(zip(axes, valid_images)):
            try:
                img_path = Path(img_data['image_path'])
                if not img_path.exists():
                    raise FileNotFoundError(f"Image file not found: {img_path}")
                
                img = PILImage.open(img_path)
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f"Page {img_data['page_number']}", fontsize=10)
            except Exception as e:
                print(f"Error loading image {img_data['image_path']}: {str(e)}")
                ax.text(0.5, 0.5, f"Error loading image\n{str(e)}", ha='center', va='center')
                ax.axis('off')

        # Remove empty subplots
        for ax in axes[num_images:]:
            ax.remove()

        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    pdf_processor = ExtractMetadata()
    test_pdf = "./pdfs/LightRAG.pdf"
    results = pdf_processor.process_pdf(test_pdf)