"""
Vision Handler Module

Handles vision tasks using Qwen3-VL for image understanding and analysis.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import base64
import logging
import re
import time
import requests
from PIL import Image
from io import BytesIO

logger = logging.getLogger(__name__)


@dataclass
class VisionRequest:
    """Request for vision analysis."""
    image: str  # Base64 encoded image or path
    prompt: str
    return_bboxes: bool = False  # Return bounding boxes for GUI automation
    max_tokens: int = 1024


@dataclass
class VisionResponse:
    """Response from vision analysis."""
    content: str
    bounding_boxes: Optional[List[Dict]] = None
    ocr_text: Optional[str] = None
    confidence: float = 1.0
    latency_ms: float = 0.0


class VisionHandler:
    """
    Qwen3-VL handler for vision tasks.

    Capabilities:
    - Image understanding and description
    - GUI automation (find buttons, menus)
    - Screenshot analysis
    - OCR (Optical Character Recognition)
    - Document understanding
    - Bounding box detection
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-7B-AWQ",
        max_image_size: Tuple[int, int] = (1024, 1024)
    ):
        """
        Initialize vision handler.

        Args:
            model_name: Vision model name
            max_image_size: Maximum image dimensions (width, height) for processing
        """
        self.model_name = model_name
        self.max_image_size = max_image_size
        self._model = None
        self._processor = None

    def load(self):
        """
        Load vision model.

        In production, this would initialize Qwen3-VL using vLLM or transformers.
        Currently uses mock implementation for development.
        """
        logger.info(f"Loading vision model: {self.model_name}")

        try:
            # Placeholder for actual model loading
            # Production implementation would use:
            #
            # Option 1: vLLM (GPU-accelerated)
            # from vllm import LLM
            # self._model = LLM(
            #     model=self.model_name,
            #     trust_remote_code=True,
            #     gpu_memory_utilization=0.9,
            # )
            #
            # Option 2: Transformers (more flexible)
            # from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            # self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            #     self.model_name,
            #     torch_dtype="auto",
            #     device_map="auto"
            # )
            # self._processor = AutoProcessor.from_pretrained(self.model_name)

            # Mock implementation for development
            self._model = "mock_vision_model"
            self._processor = "mock_processor"

            logger.info(f"Vision model loaded: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load vision model: {e}")
            raise RuntimeError(f"Vision model loading failed: {e}")

    async def analyze(self, request: VisionRequest) -> VisionResponse:
        """
        Analyze image with Qwen3-VL.

        Args:
            request: Vision request with image and prompt

        Returns:
            Vision response with analysis
        """
        return await self.analyze_image(
            image=request.image,
            prompt=request.prompt,
            return_bboxes=request.return_bboxes
        )

    async def analyze_image(
        self,
        image: str,
        prompt: str = "Describe this image in detail",
        return_bboxes: bool = False
    ) -> VisionResponse:
        """
        Analyze image with Qwen3-VL.

        Args:
            image: Base64-encoded image or URL
            prompt: Question/instruction about the image
            return_bboxes: Whether to detect and return bounding boxes

        Returns:
            VisionResponse with description, boxes, OCR text
        """
        start_time = time.time()

        try:
            # 1. Preprocess image
            img = self._preprocess_image(image)
            logger.info(f"Image preprocessed: {img.size} {img.mode}")

            # 2. Format prompt for Qwen-VL
            if return_bboxes:
                prompt = f"{prompt} Provide bounding boxes for objects."

            # Check if OCR is requested
            ocr_requested = any(keyword in prompt.lower() for keyword in ['ocr', 'text', 'read', 'extract'])

            # 3. Run inference
            # In production, would use Qwen-VL's image-text format:
            # <img>image_tensor</img>User: {prompt}\nAssistant:

            if self._model == "mock_vision_model":
                # Mock implementation for development
                response_text = self._generate_mock_response(prompt, return_bboxes, ocr_requested)
            else:
                # Production implementation would be:
                # from vllm import SamplingParams
                #
                # # Prepare inputs
                # messages = [
                #     {"role": "user", "content": f"<img>{img}</img>{prompt}"}
                # ]
                #
                # sampling_params = SamplingParams(
                #     temperature=0.7,
                #     max_tokens=1024,
                # )
                #
                # outputs = self._model.generate(messages, sampling_params)
                # response_text = outputs[0].outputs[0].text
                response_text = "[Production model not loaded]"

            # 4. Parse output
            description = response_text
            bounding_boxes = None
            ocr_text = None

            if return_bboxes:
                bounding_boxes = self._parse_bounding_boxes(response_text)
                # Remove bbox tags from description
                description = re.sub(r'<ref>.*?</ref><box>.*?</box>', '', response_text).strip()

            if ocr_requested:
                ocr_text = self._extract_ocr_text(response_text)

            # 5. Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            logger.info(f"Vision analysis complete in {latency_ms:.2f}ms")

            return VisionResponse(
                content=description,
                bounding_boxes=bounding_boxes,
                ocr_text=ocr_text,
                latency_ms=latency_ms
            )

        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            latency_ms = (time.time() - start_time) * 1000

            # Return error response
            return VisionResponse(
                content=f"Error: {str(e)}",
                bounding_boxes=None,
                ocr_text=None,
                confidence=0.0,
                latency_ms=latency_ms
            )

    def _preprocess_image(self, image: str) -> Image.Image:
        """
        Preprocess image from base64 or URL.

        Steps:
        1. Decode base64 or fetch URL
        2. Convert to PIL Image
        3. Resize if larger than max_image_size
        4. Convert to RGB

        Args:
            image: Base64-encoded image or URL

        Returns:
            PIL Image object

        Raises:
            ValueError: If image is invalid or unsupported format
        """
        try:
            # Detect if base64 or URL
            if image.startswith('http://') or image.startswith('https://'):
                # Fetch from URL
                logger.info(f"Fetching image from URL")
                response = requests.get(image, timeout=10)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content))
            else:
                # Decode base64
                # Handle data:image/png;base64,... format
                if ',' in image and image.startswith('data:'):
                    image = image.split(',', 1)[1]

                try:
                    img_data = base64.b64decode(image)
                    img = Image.open(BytesIO(img_data))
                except Exception as e:
                    raise ValueError(f"Invalid base64 image data: {e}")

            # Verify it's a valid image
            img.verify()

            # Reopen after verify (verify() closes the file)
            if image.startswith('http://') or image.startswith('https://'):
                response = requests.get(image, timeout=10)
                img = Image.open(BytesIO(response.content))
            else:
                if ',' in image and image.startswith('data:'):
                    image = image.split(',', 1)[1]
                img_data = base64.b64decode(image)
                img = Image.open(BytesIO(img_data))

            # Resize if needed
            if img.size[0] > self.max_image_size[0] or img.size[1] > self.max_image_size[1]:
                logger.info(f"Resizing image from {img.size} to fit {self.max_image_size}")
                img.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)

            # Convert to RGB (handle RGBA, grayscale, etc.)
            if img.mode != 'RGB':
                logger.info(f"Converting image from {img.mode} to RGB")
                if img.mode == 'RGBA':
                    # Create white background for transparency
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
                    img = background
                else:
                    img = img.convert('RGB')

            return img

        except requests.RequestException as e:
            raise ValueError(f"Failed to fetch image from URL: {e}")
        except Exception as e:
            raise ValueError(f"Failed to process image: {e}")

    def _parse_bounding_boxes(self, text: str) -> List[Dict]:
        """
        Parse bounding boxes from Qwen-VL output.

        Qwen-VL format: <ref>cat</ref><box>[[x1,y1,x2,y2]]</box>

        Args:
            text: Model output text containing bounding boxes

        Returns:
            List of {"label": "cat", "box": [x1, y1, x2, y2]}
        """
        boxes = []

        # Extract <ref>...</ref><box>...</box> pairs
        # Qwen-VL uses normalized coordinates [0-1000]
        pattern = r'<ref>(.*?)</ref><box>\[\[(\d+),(\d+),(\d+),(\d+)\]\]</box>'
        matches = re.findall(pattern, text)

        for match in matches:
            label, x1, y1, x2, y2 = match
            boxes.append({
                "label": label,
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": 0.95  # Placeholder confidence
            })

        logger.info(f"Parsed {len(boxes)} bounding boxes")
        return boxes

    def _extract_ocr_text(self, text: str) -> str:
        """
        Extract OCR text from model response.

        Args:
            text: Model output text

        Returns:
            Extracted text content
        """
        # In production, Qwen-VL would directly output OCR text
        # For mock, extract text between common markers

        # Remove bounding box tags
        clean_text = re.sub(r'<ref>.*?</ref><box>.*?</box>', '', text)

        # Extract quoted text or return as-is
        return clean_text.strip()

    def _generate_mock_response(
        self,
        prompt: str,
        return_bboxes: bool,
        ocr_requested: bool
    ) -> str:
        """
        Generate mock response for development/testing.

        Args:
            prompt: User prompt
            return_bboxes: Whether to include bounding boxes
            ocr_requested: Whether OCR was requested

        Returns:
            Mock response text
        """
        response_parts = []

        # Generate description based on prompt
        if "screenshot" in prompt.lower() or "gui" in prompt.lower():
            response_parts.append("This is a screenshot showing a desktop application interface with multiple UI elements.")
        elif "document" in prompt.lower():
            response_parts.append("This appears to be a document with formatted text and possibly tables or diagrams.")
        elif "ocr" in prompt.lower() or "text" in prompt.lower():
            response_parts.append("Detected text content in the image: Sample text from the document.")
        else:
            response_parts.append("This image shows various objects and scenes as described in the analysis.")

        # Add bounding boxes if requested
        if return_bboxes:
            response_parts.append(
                "<ref>button</ref><box>[[100,200,200,250]]</box>"
                "<ref>text field</ref><box>[[100,300,400,350]]</box>"
                "<ref>menu</ref><box>[[50,50,150,100]]</box>"
            )

        # Add OCR text if requested
        if ocr_requested:
            response_parts.append("\n\nExtracted Text:\nSample document text\nLine 1\nLine 2")

        return " ".join(response_parts)

    async def analyze_screenshot(
        self,
        image: str,
        task: str
    ) -> VisionResponse:
        """
        Analyze screenshot for GUI automation.

        Args:
            image: Screenshot image (base64 or path)
            task: Task description (e.g., "find the save button")

        Returns:
            Vision response with element locations
        """
        request = VisionRequest(
            image=image,
            prompt=f"Screenshot analysis: {task}",
            return_bboxes=True
        )

        return await self.analyze(request)

    async def ocr(self, image: str) -> str:
        """
        Extract text from image using OCR.

        Args:
            image: Image to extract text from

        Returns:
            Extracted text
        """
        request = VisionRequest(
            image=image,
            prompt="Extract all text from this image. Maintain formatting where possible."
        )

        response = await self.analyze(request)
        return response.ocr_text or response.content

    async def describe_image(self, image: str) -> str:
        """
        Generate detailed description of image.

        Args:
            image: Image to describe

        Returns:
            Image description
        """
        request = VisionRequest(
            image=image,
            prompt="Provide a detailed description of this image, including objects, people, setting, and any text visible."
        )

        response = await self.analyze(request)
        return response.content

    def unload(self):
        """Unload vision model and free resources."""
        logger.info(f"Unloading vision model: {self.model_name}")
        self._model = None
        self._processor = None
