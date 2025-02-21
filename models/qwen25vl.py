import tempfile

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers.generation import GenerationConfig
import json
import re
import os
from PIL import Image

from qwen_vl_utils import process_vision_info
# added from cookbook
import json
from PIL import Image
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize

from qwen25_agent_function_call import ComputerUse


"""def perform_gui_grounding(screenshot_path, user_query, model, processor):

    # Open and process image
    input_image = Image.open(screenshot_path)
    resized_height, resized_width = smart_resize(
        input_image.height,
        input_image.width,
        factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
        min_pixels=processor.image_processor.min_pixels,
        max_pixels=processor.image_processor.max_pixels,
    )

    # Initialize computer use function
    computer_use = ComputerUse(
        cfg={"display_width_px": resized_width, "display_height_px": resized_height}
    )

    # Build messages
    message = NousFnCallPrompt.preprocess_fncall_messages(
        messages=[
            Message(role="system", content=[ContentItem(text="You are a helpful assistant.")]),
            Message(role="user", content=[
                ContentItem(text=user_query),
                ContentItem(image=f"file://{screenshot_path}")
            ]),
        ],
        functions=[computer_use.function],
        lang=None,
    )
    message = [msg.model_dump() for msg in message]

    # Process input
    text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[input_image], padding=True, return_tensors="pt").to('cuda')

    # Generate output
    output_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

    # Parse action and visualize
    action = json.loads(output_text.split('<tool_call>\n')[1].split('\n</tool_call>')[0])
    bbox = action['arguments']['coordinates']
    #display_image = input_image.resize((resized_width, resized_height))
    #display_image = draw_point(input_image, action['arguments']['coordinate'], color='green')

    return output_text
# added from cookbook
"""

# bbox -> point (str)
def bbox_2_point(bbox, dig=2):
    # bbox [left, top, right, bottom]
    point = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
    point = [f"{item:.2f}" for item in point]
    point_str = "({},{})".format(point[0], point[1])
    return point_str

# bbox -> bbox (str)
def bbox_2_bbox(bbox, dig=2):
    bbox = [f"{item:.2f}" for item in bbox]
    bbox_str = "({},{},{},{})".format(bbox[0], bbox[1], bbox[2], bbox[3])
    return bbox_str

# point (str) -> point
def pred_2_point(s):
    floats = re.findall(r'-?\d+\.?\d*', s)
    floats = [float(num) for num in floats]
    if len(floats) == 2:
        return floats
    elif len(floats) == 4:
        return [(floats[0]+floats[2])/2, (floats[1]+floats[3])/2]
    else:
        return None

# bbox (qwen str) -> bbox
def extract_bbox(s):
    pattern = r"<\|box_start\|\>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|\>"
    matches = re.findall(pattern, s)
    if matches:
        # Get the last match and return as tuple of integers
        last_match = matches[-1]
        return (int(last_match[0]), int(last_match[1])), (int(last_match[2]), int(last_match[3]))
    return None


def image_to_temp_filename(image):
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(temp_file.name)
    print(f"Image saved to temporary file: {temp_file.name}")
    return temp_file.name


class Qwen25VLModel():
    def load_model(self, model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct"):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path, 
            device_map="cuda", 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

        # Setting default generation config
        self.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True).to_dict()
        self.set_generation_config(
            max_length=2048,
            do_sample=False,
            temperature=0.0
        )

    def set_generation_config(self, **kwargs):
        self.generation_config.update(**kwargs)
        self.model.generation_config = GenerationConfig(**self.generation_config)

    def ground_only_positive(self, instruction, image):
        """
        Ground user instruction on an image using the new model format.
        Returns coordinates in the original format for compatibility.
        """
        # Handle image input
        if not isinstance(image, str):
            assert isinstance(image, Image.Image)
            image_path = image_to_temp_filename(image)
        else:
            image_path = image
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."

        # Open and process image for resizing
        input_image = Image.open(image_path)
        resized_height, resized_width = smart_resize(
            input_image.height,
            input_image.width,
            factor=self.processor.image_processor.patch_size * self.processor.image_processor.merge_size,
            min_pixels=self.processor.image_processor.min_pixels,
            max_pixels=self.processor.image_processor.max_pixels,
        )

        # Initialize computer use function
        computer_use = ComputerUse(
            cfg={"display_width_px": resized_width, "display_height_px": resized_height}
        )

        # Build messages using cookbook format
        message = NousFnCallPrompt.preprocess_fncall_messages(
            messages=[
                Message(role="system", content=[ContentItem(text="You are a helpful assistant.")]),
                Message(role="user", content=[
                    ContentItem(text=instruction),
                    ContentItem(image=f"file://{image_path}")
                ]),
            ],
            functions=[computer_use.function],
            lang=None,
        )
        message = [msg.model_dump() for msg in message]

        # Process input
        text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=[input_image],
            padding=True,
            return_tensors="pt"
        ).to('cuda')

        # Generate output
        output_ids = self.model.generate(**inputs, max_new_tokens=2048)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False
        )[0]

        # Initialize result dictionary with default values
        result_dict = {
            "result": None,
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }

        try:
            # Extract coordinates from tool call format
            if '<tool_call>' in response and '</tool_call>' in response:
                tool_call = response.split('<tool_call>\n')[1].split('\n</tool_call>')[0]
                action_dict = json.loads(tool_call)

                if 'arguments' in action_dict and 'coordinate' in action_dict['arguments']:
                    # Get coordinates and normalize to 0-1 range
                    coords = action_dict['arguments']['coordinate']
                    normalized_x = coords[0] / resized_width
                    normalized_y = coords[1] / resized_height

                    # Set point in result dict
                    result_dict["point"] = [normalized_x, normalized_y]

                    # Create bbox from point (small box around click point)
                    box_size = 0.02  # 2% of image size
                    result_dict["bbox"] = [
                        max(0, normalized_x - box_size),
                        max(0, normalized_y - box_size),
                        min(1, normalized_x + box_size),
                        min(1, normalized_y + box_size)
                    ]

        except Exception as e:
            print(f"Error parsing response: {e}")
            print('---------------')
            print(response)

        # set result status
        if result_dict["bbox"] or result_dict["point"]:
            result_status = "positive"
        elif "Target does not exist".lower() in response.lower():
            result_status = "negative"
        else:
            result_status = "wrong_format"
        result_dict["result"] = result_status

        return result_dict

    def ground_allow_negative(self, instruction, image):
        """
        Ground user instruction on an image using the new model format.
        Allows for negative responses when target does not exist.
        Returns coordinates in the original format for compatibility.
        """
        # Handle image input
        if not isinstance(image, str):
            assert isinstance(image, Image.Image)
            image_path = image_to_temp_filename(image)
        else:
            image_path = image
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."

        # Open and process image for resizing
        input_image = Image.open(image_path)
        resized_height, resized_width = smart_resize(
            input_image.height,
            input_image.width,
            factor=self.processor.image_processor.patch_size * self.processor.image_processor.merge_size,
            min_pixels=self.processor.image_processor.min_pixels,
            max_pixels=self.processor.image_processor.max_pixels,
        )

        # Initialize computer use function
        computer_use = ComputerUse(
            cfg={"display_width_px": resized_width, "display_height_px": resized_height}
        )

        # Build messages using cookbook format
        message = NousFnCallPrompt.preprocess_fncall_messages(
            messages=[
                Message(role="system", content=[ContentItem(text="You are a helpful assistant.")]),
                Message(role="user", content=[
                    ContentItem(
                        text=f"{instruction}. If the target does not exist, respond with 'Target does not exist'."),
                    ContentItem(image=f"file://{image_path}")
                ]),
            ],
            functions=[computer_use.function],
            lang=None,
        )
        message = [msg.model_dump() for msg in message]

        # Process input
        text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=[input_image],
            padding=True,
            return_tensors="pt"
        ).to('cuda')

        # Generate output
        output_ids = self.model.generate(**inputs, max_new_tokens=2048)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False
        )[0]

        # Initialize result dictionary with default values
        result_dict = {
            "result": None,
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }

        try:
            # Extract coordinates from tool call format
            if '<tool_call>' in response and '</tool_call>' in response:
                tool_call = response.split('<tool_call>\n')[1].split('\n</tool_call>')[0]
                action_dict = json.loads(tool_call)

                if 'arguments' in action_dict and 'coordinate' in action_dict['arguments']:
                    # Get coordinates and normalize to 0-1 range
                    coords = action_dict['arguments']['coordinate']
                    normalized_x = coords[0] / resized_width
                    normalized_y = coords[1] / resized_height

                    # Set point in result dict
                    result_dict["point"] = [normalized_x, normalized_y]

                    # Create bbox from point (small box around click point)
                    box_size = 0.02  # 2% of image size
                    result_dict["bbox"] = [
                        max(0, normalized_x - box_size),
                        max(0, normalized_y - box_size),
                        min(1, normalized_x + box_size),
                        min(1, normalized_y + box_size)
                    ]

        except Exception as e:
            print(f"Error parsing response: {e}")
            print('---------------')
            print(response)

        # set result status
        if result_dict["bbox"] or result_dict["point"]:
            result_status = "positive"
        elif "Target does not exist".lower() in response.lower():
            result_status = "negative"
        else:
            result_status = "wrong_format"
        result_dict["result"] = result_status

        return result_dict

