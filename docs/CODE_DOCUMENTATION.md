# Code Documentation

## Notebook Structure

### Preprocessing (Cells 4-7)
- XML parsing from MedQuAD
- Data cleaning and filtering
- Train/val/test splitting
- Instruction formatting

### Training (Cells 8-13)
- Model setup with LoRA
- 3 hyperparameter configurations
- Training loops
- Results comparison

### Evaluation (Cells 14-17)
- Baseline evaluation
- Fine-tuned evaluation
- Statistical comparison
- Error analysis

### Deployment (Cell 18)
- Gradio interface
- HuggingFace Spaces deployment

## Key Functions

**MedQuADPreprocessor**: Handles all data preprocessing
**generate_response()**: Inference function
**answer_medical_question()**: Gradio interface function

See inline comments in notebook for detailed explanations.
