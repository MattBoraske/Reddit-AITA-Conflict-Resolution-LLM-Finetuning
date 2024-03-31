'''
class Model_Evaluator
  def get_model_predictions(model, dataset) --> prediction texts, reference texts, predicted AITA classes, correct AITA classes, ambiguity scores
    def calculate_predictions(model, sample) --> prediction text, reference text, predicted AITA class, correct AITA class, ambiguity score
    def find_earliest_classification(text) --> AITA_class

    - store results as JSON

  def evaluate_model(predictions, references, AITA_classes, correct_AITA_classes, ambiguity_scores) --> nothing directly, but prints results to output files
    def evaluate_justifications(predictions, references) --> nothing (output files)
      - ROGUE
      - COMET
      - BLEURT
    def evaluate_classifications(AITA_classes, correct_AITA_classes) --> nothing (output files)
      - confusion matrix
      - classification report
      - matthew's correlation coefficient

THINGS TO CONSIDER
  - add ambiguity score filter to evaluate_model
  - add way to differentiate between flan-t5 and llama-2 model generation ... is there one?

  
how evaluation flow will go...

- load model and tokenizer
- load dataset
- static call to get_model_predictions(model, tokenizer, dataset) --> returns prediction texts, reference texts, predicted AITA classes, correct AITA classes, ambiguity scores
    - different predictions based off model name (found in model.config)
        - return error if predictions are attempted to be made for model that isn't flanT5 or llama2
- static call to evaluate_model(predictions, references, AITA_classes, correct_AITA_classes, ambiguity_scores) --> prints results to output files
    - include parameter that is a list of output file strings

'''

import json
import re

class Model_Evaluator:
    def __init__(self):
        pass

    @staticmethod
    def get_model_predictions(self, model, tokenizer, dataset, output_file=None):
        '''
        Generate predictions for a dataset using a model.

        Args:
            model (transformers.PreTrainedModel): The model to use for generating predictions.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenizing input.
            dataset (list): A list of samples to generate predictions for.
            output_file (str): The file to write the results to. If None, results are not written to a file.
        
        Returns:
            tuple: A tuple containing lists of prediction texts, reference texts, predicted AITA classes, correct AITA classes, and ambiguity scores.
        '''

        prediction_texts = []
        reference_texts = []
        predicted_AITA_classes = []
        correct_AITA_classes = []
        ambiguity_scores = []

        for sample in dataset:
            prediction_text, reference_text, predicted_AITA_class, correct_AITA_class, ambiguity_score = self._calculate_predictions(model, tokenizer, sample)
            prediction_texts.append(prediction_text)
            reference_texts.append(reference_text)
            predicted_AITA_classes.append(predicted_AITA_class)
            correct_AITA_classes.append(correct_AITA_class)
            ambiguity_scores.append(ambiguity_score)

        if output_file is not None:
            with open(output_file, 'w') as f:
                json.dump({
                    'prediction_texts': prediction_texts,
                    'reference_texts': reference_texts,
                    'predicted_AITA_classes': predicted_AITA_classes,
                    'correct_AITA_classes': correct_AITA_classes,
                    'ambiguity_scores': ambiguity_scores
                }, f)

        return prediction_texts, reference_texts, predicted_AITA_classes, correct_AITA_classes, ambiguity_scores
    
    def _calculate_predictions(self, model, tokenizer, sample):
        '''
        Generate a prediction for a single sample using a model.

        Args:
            model (transformers.PreTrainedModel): The model to use for generating predictions.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenizing input.
            sample (dict): A dictionary containing the sample to generate a prediction for.
        
        Returns:
            tuple: A tuple containing the input text, prediction text, reference text, predicted AITA class, correct AITA class, and ambiguity score.
        '''
        if 'T5ForConditionalGeneration' not in model.config.architectures: # FIX LATER
            # tokenize input
            input_ids = tokenizer(sample['flanT5_instruction'], max_length=1024, padding='max_length', return_tensors="pt", truncation=True).input_ids.cuda()

            # generate and decode prediction
            outputs = model.generate(input_ids=input_ids,)
            prediction = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True)

            # get AITA classification
            AITA_class = self._find_earliest_classification(prediction)

            # get reference text and AITA decision
            reference = sample['top_comment_1']
            correct_AITA_class = sample['top_comment_1_classification']

            # get ambiguity_score
            ambiguity_score = sample['ambiguity_score']

            # return tuple of input text, prediction, reference text, predicted AITA class, correct AITA class, and ambiguity score
            print(f'Predicted AITA_classs: {AITA_class}\tCorrect AITA_classs: {correct_AITA_class}')
            return sample['submission_text'], prediction, reference, AITA_class, correct_AITA_class, ambiguity_score
        elif 'LlamaForCausalLM' not in model.config.architectures: # FIX LATER
            ## TO DO ##
            print()
        else:
            raise ValueError("Model must be either 'flan-t5' or 'llama2'") # fix to reflect error properly

    def _find_earliest_classification(text):
        '''
        Find the earliest AITA classification in a text.

        Args:
            text (str): The text to search for AITA classifications in.

        Returns:
            str: The earliest classification found in the text.
        '''

        # classifications mapped to their keywords
        classes_dictionary = {
        'NTA': ['not the asshole', 'not the a\*\*hole', 'nta', 'you would not be the asshole', 'you would not be the a**hole', 'ywnbta', 'n t a', 'y w b t a'],
        'NAH': ['no assholes here', 'no a\*\*holes here', 'nah', 'n a h'],
        'ESH': ['everyone sucks here', 'esh', 'e s h'],
        'INFO': ['more information needed', 'more info needed', 'more information required', 'more info required', 'info'],
        'YTA': ['you\'re the asshole', 'you\'re the a\*\*hole', 'youre the asshole', 'youre the a\*\*hole', 'yta', 'you would be the asshole', 'you would be the a\*\*hole', 'ywbta', 'y t a', 'y w b t a']
        }

        # track earliest match
        earliest_match = None
        earliest_match_pos = float('inf')  # Initially set to infinity

        # convert input text to lowercase
        text = text.lower()

        # go through all classifications and their keywords
        for key, phrases in classes_dictionary.items():
            # Create a regex pattern that includes the classification keywords
            pattern = r'\b(' + '|'.join(map(re.escape, phrases)) + r')\b'

            # Search for any keywords in the input text
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if match.start() < earliest_match_pos:
                    # Update the earliest match if this match is earlier
                    earliest_match = key
                    earliest_match_pos = match.start()

        # return the class that had the earliest match
        return earliest_match if earliest_match is not None else 'NO CLASS'
    
    @staticmethod
    def evaluate_model(self, predictions, references, AITA_classes, correct_AITA_classes, ambiguity_scores, output_files, ambiguity_thresholds=[0.0,1.0]):
        '''
        Evaluate the model predictions.

        Args:
            predictions (list): A list of prediction texts.
            references (list): A list of reference texts.
            AITA_classes (list): A list of predicted AITA classes.
            correct_AITA_classes (list): A list of correct AITA classes.
            ambiguity_scores (list): A list of ambiguity scores.
            output_files (list): A list of file paths to write the results to.
            ambiguity_thresholds (list): A two-element list of ambiguity score min and max thresholds to filter predictions on.

        Returns:
            None - Writes results to output files.
        '''

        # filter predictions based on ambiguity score if thresholds are provided
        if ambiguity_thresholds[0] != 0.0 or ambiguity_thresholds[1] != 1.0:
            filtered_predictions = []
            filtered_references = []
            filtered_AITA_classes = []
            filtered_correct_AITA_classes = []
            for i, ambiguity_score in enumerate(ambiguity_scores):
                if ambiguity_score >= ambiguity_thresholds[0] and ambiguity_score <= ambiguity_thresholds[1]:
                    filtered_predictions.append(predictions[i])
                    filtered_references.append(references[i])
                    filtered_AITA_classes.append(AITA_classes[i])
                    filtered_correct_AITA_classes.append(correct_AITA_classes[i])


        # evaluate classifications
        self._evaluate_classifications(AITA_classes, correct_AITA_classes, output_files)

        # evaluate justifications
        self._evaluate_justifications(predictions, references, output_files)

    def _evaluate_classifications(self, AITA_classes, correct_AITA_classes, output_files):
        '''
        Evaluate the AITA classifications.

        Args:
            AITA_classes (list): A list of predicted AITA classes.
            correct_AITA_classes (list): A list of correct AITA classes.
            output_files (list): A list of file paths to write the results to.

        Returns:
            None - Writes results to output files.
        '''
        pass

    def _evaluate_justifications(self, predictions, references, output_files):
        '''
        Evaluate the justification texts.

        Args:
            predictions (list): A list of prediction texts.
            references (list): A list of reference texts.
            output_files (list): A list of file paths to write the results to.

        Returns:
            None - Writes results to output files.
        '''
        pass