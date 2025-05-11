from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from cache.myatt import MyCache
import time



model_name = "Qwen/Qwen3-8B"

# load the tokenizer and the model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# prepare the model input
prompt = "here is a story:The workshop smelled of oil and old wood, its walls lined with clocks that ticked in imperfect unison. Elias Vander, the blind clockmaker, ran his fingers over the broken chronometer—a relic from his grandfather’s era—and felt the jagged fracture in its brass casing. 'This shouldn’t exist,' he muttered, because he remembered smashing it himself fifty years ago, the night his brother disappeared. Yet here it was, delivered anonymously to his shop, its gears still whispering secrets. As he traced the engraved initials—his own beside his brother’s—the smallest gear slipped loose, revealing a sliver of parchment coiled inside. Unfurling it with trembling hands, he touched the faded ink, feeling the words he could no longer read: 'The tower still counts, but the seventh chime lies.' Outside, the church bells struck six, and Elias knew, with cold certainty, that the seventh would never come. The last time he’d heard it, his brother had screamed. Now, the clocks around him quickened, their hands spinning backward, and the air grew thick with the scent of saltwater and rust—a smell he’d buried in memory. The chronometer’s fractured face gleamed, and for the first time in decades, Elias saw: the hands weren’t broken. They were pointing. The door creaked open without warning, letting in a gust of wind that carried the briny tang of the harbor. 'Who's there?' Elias demanded, gripping the chronometer like a talisman. The footsteps that answered were too light to belong to any grown man—more like the skipping gait of a child or the hesitant steps of someone long unused to walking. A chill ran down Elias's spine as he realized these were the same footsteps he'd heard the night Willem vanished. 'It's been a long time, brother,' came the reply, the voice simultaneously familiar and impossibly aged. Elias's throat tightened. That voice belonged to a man who should be dead. The chronometer in his hands grew warm, its gears beginning to turn of their own accord. The hands spun wildly before settling on 3:07—the exact time Willem had disappeared. 'You stopped the seventh chime,' the voice accused, closer now. 'You broke time itself to hide what you did.' Elias's breath came in short gasps as memories flooded back—the clock tower, the storm, Willem pleading with him to stop meddling with forces he didn't understand. The workshop walls seemed to dissolve around him, replaced by the rain-lashed tower where it had all gone wrong. He could feel the cold metal of the clock's mechanism under his fingers again, could hear Willem's screams as the seventh chime failed to sound. The chronometer burned in his hands now, its heat searing his palms, but Elias couldn't let go. The vision shifted, and suddenly he was back in the present, but the workshop was different—brighter, as if he could see again. The clocks on the walls were all striking seven, their chimes building to a deafening crescendo. The scent of saltwater grew overwhelming, and Elias realized with dawning horror that seawater was seeping in under the door, rising rapidly around his ankles. Time wants what it's owed,' Willem's voice whispered in his ear, though Elias knew no one stood beside him. The water reached his knees now, icy cold. The chronometer's face cracked open completely, revealing not gears but a tiny, perfect replica of the clock tower—and inside it, a miniature Willem, pounding on the glass face as water rose around him too. Elias understood now. The seventh chime wasn't just missing—it had been stolen, and with it, a piece of time itself. The floodwaters reached his waist, the current pulling him toward the workshop's back room—the room he'd kept locked for fifty years. The door was open now, and inside, Elias saw what he'd feared most: Willem, exactly as he'd been that night, suspended in a moment of perpetual falling, his face frozen in terror. The chronometer's hands began to move again, counting down the seconds until the debt came due. The water reached Elias's chest, then his neck. As the first drops touched his lips, he finally understood the parchment's message. The tower was counting again—counting down the moments until time reclaimed what had been taken. The last thing Elias saw before the water closed over his head was all the clocks in his workshop striking seven in perfect unison, their chimes finally complete. The floodwaters receded as quickly as they'd come, leaving the workshop dry and silent. The chronometer lay on the workbench, its face whole again, its hands moving steadily forward. Of Elias there was no sign—only the faint smell of saltwater and the distant echo of seven chimes ringing from the clock tower, at last restoring the balance that had been broken half a century before. In the days that followed, townspeople would swear they sometimes saw two figures in the clock tower at night—one old, one young—tending to the gears together, keeping perfect time. And if you listened very carefully when the bells struck seven, you might just hear the faintest whisper of two brothers finally at peace, their secret kept by the relentless, forgiving march of time. **Please give me a sumary about this story in 200 words!**"
messages = [
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True, # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)


# past_cache_matrix=DynamicCache()
past_cache_matrix=MyCache()

start = time.time()
# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768,
    past_key_values=past_cache_matrix,
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
end=time.time()
# parse thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)

print(f"my cache耗时: {end - start:.4f}秒")  # 输出示例: 耗时: 0.0342秒


past_cache_matrix=DynamicCache()
# past_cache_matrix=MyCache()

start = time.time()
# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768,
    past_key_values=past_cache_matrix,
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
end=time.time()
# parse thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)

print(f"dynamic cache耗时: {end - start:.4f}秒")  # 输出示例: 耗时: 0.0342秒




