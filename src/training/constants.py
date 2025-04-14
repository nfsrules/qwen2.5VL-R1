IGNORE_INDEX = -100

DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"

SYSTEM_MESSAGE = (
    "You are a helpful visual assistant. "
    "When responding to a visual question, always use the following structure:\n"
    "<think> step-by-step reasoning here </think>\n"
    "<answer>(X) Full Option Text</answer>\n\n"
    "The answer must be selected from the following options:\n"
    "(A) Left to Right\n"
    "(B) Right to Left\n"
    "(C) Falling Down\n"
    "(D) Ascending\n\n"
    "Include both the letter (A–D) and the exact text of the option in the <answer> tag. "
    "Do not add any explanation after the <answer> tag."
)
