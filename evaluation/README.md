# Evaluation files
## Running llava code
`python test.py --path <src_data_path> --prompt_label <labelled or labelled_id> --output_dir <output_dir_path>`

eg. `python test.py --path ../../data/depth_synthetic_2D/images-3-shapes --prompt_label labelled_id --output_dir ../../outputs`
## Running GPT-v code
The model calling api can be replaced with openai package and API key.
- `test.py` : Evaluates List and MCQ questions on GPT-v
- `test-tf.py`: Evaluates T/F on GPT-v
