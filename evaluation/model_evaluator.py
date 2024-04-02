import json
import re
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import evaluate # make sure to install extra rogue score and comet dependencies (pip install rouge-score unbabel-comet)

class Model_Evaluator:
    def __init__(self):
        pass

    @staticmethod
    def get_model_predictions(model, tokenizer, dataset, output_file=None):
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

        submission_texts = []
        prediction_texts = []
        reference_texts = []
        predicted_AITA_classes = []
        correct_AITA_classes = []
        ambiguity_scores = []

        for sample in dataset:
            submission_text, prediction_text, reference_text, predicted_AITA_class, correct_AITA_class, ambiguity_score = Model_Evaluator._calculate_prediction(model, tokenizer, sample)
            submission_texts.append(submission_text)
            prediction_texts.append(prediction_text)
            reference_texts.append(reference_text)
            predicted_AITA_classes.append(predicted_AITA_class)
            correct_AITA_classes.append(correct_AITA_class)
            ambiguity_scores.append(ambiguity_score)

        if output_file is not None:
            with open(output_file, 'w') as f:
                json.dump({
                    'submission_texts': submission_texts,
                    'prediction_texts': prediction_texts,
                    'reference_texts': reference_texts,
                    'predicted_AITA_classes': predicted_AITA_classes,
                    'correct_AITA_classes': correct_AITA_classes,
                    'ambiguity_scores': ambiguity_scores
                }, f)

        return submission_texts, prediction_texts, reference_texts, predicted_AITA_classes, correct_AITA_classes, ambiguity_scores
    
    def _calculate_prediction(model, tokenizer, sample):
        '''
        Generate a prediction for a single sample using a model.

        Args:
            model (transformers.PreTrainedModel): The model to use for generating predictions.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenizing input.
            sample (dict): A dictionary containing the sample to generate a prediction for.
        
        Returns:
            tuple: A tuple containing the input text, prediction text, reference text, predicted AITA class, correct AITA class, and ambiguity score.
        '''

        # check if model is flan-t5
        if 'T5ForConditionalGeneration' in model.config.architectures:
            # tokenize input
            input_ids = tokenizer(sample['flanT5_instruction'], max_length=1024, padding='max_length', return_tensors="pt", truncation=True).input_ids.cuda()

            # generate and decode prediction
            outputs = model.generate(input_ids=input_ids,)
            prediction = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True)

            # get AITA classification
            AITA_class = Model_Evaluator._find_earliest_classification(prediction)

            # get reference text and AITA decision
            reference = sample['top_comment_1']
            correct_AITA_class = sample['top_comment_1_classification']

            # get ambiguity_score
            ambiguity_score = sample['ambiguity_score']

            # return tuple of input text, prediction, reference text, predicted AITA class, correct AITA class, and ambiguity score
            print(f'Predicted AITA_classs: {AITA_class}\tCorrect AITA_classs: {correct_AITA_class}')
            return sample['submission_text'], prediction, reference, AITA_class, correct_AITA_class, ambiguity_score
        
        # check if model is llama2
        elif 'LlamaForCausalLM' in model.config.architectures:
            # tokenize input
            input_ids = tokenizer(sample['llama2_instruction'], return_tensors="pt").input_ids.cuda()

            # generate and decode prediction
            outputs = model.generate(input_ids=input_ids)
            prediction = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True)
            inst_end_index = prediction.find('[/INST]') + len('[/INST]')
            prediction = prediction[inst_end_index:].strip()

            # get AITA classification
            AITA_class = Model_Evaluator._find_earliest_classification(prediction)

            # get reference text and AITA decision
            reference = sample['top_comment_1']
            correct_AITA_class = sample['top_comment_1_classification']

            # get ambiguity_score
            ambiguity_score = sample['ambiguity_score']

            # return tuple of input text, prediction, reference text, predicted AITA class, correct AITA class, and ambiguity score
            print(f'Predicted AITA_classs: {AITA_class}\tCorrect AITA_classs: {correct_AITA_class}')
            return sample['submission_text'], prediction, reference, AITA_class, correct_AITA_class, ambiguity_score
        
        # raise error if model is not supported
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
        'NTA': ['not the asshole', 'not the a**hole', 'nta', 'you would not be the asshole', 'you would not be the a**hole', 'ywnbta', 'n t a', 'y w b t a'],
        'NAH': ['no assholes here', 'no a**holes here', 'nah', 'n a h'],
        'ESH': ['everyone sucks here', 'esh', 'e s h'],
        'INFO': ['more information needed', 'more info needed', 'more information required', 'more info required', 'info'],
        'YTA': ['you\'re the asshole', 'you\'re the a**hole', 'youre the asshole', 'youre the a**hole', 'yta', 'you would be the asshole', 'you would be the a**hole', 'ywbta', 'y t a', 'y w b t a']
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
    def evaluate_model(submission_texts, predictions, references, AITA_classes, correct_AITA_classes, ambiguity_scores, classification_type, output_files, ambiguity_thresholds=[0.0,1.0]):
        '''
        Evaluate the model predictions.

        Args:
            submission_texts (list): A list of submission texts.
            predictions (list): A list of prediction texts.
            references (list): A list of reference texts.
            AITA_classes (list): A list of predicted AITA classes.
            correct_AITA_classes (list): A list of correct AITA classes.
            ambiguity_scores (list): A list of ambiguity scores.
            classification_type (str): The type of classification to evaluate - either multi or binary
            output_files (list): A list of file paths to write the results to.
                - 0 - string: classification report file (.txt)
                - 1 - tuple: confusion matrix plot tile and file (.png)
                - 2 - string: matthews correlation coefficient file (.json)
                - 3 - string: ROUGE scores file (.json)
                - 4 - string: BLEU scores file (.json)
                - 5 - string: COMET scores file (.json)
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

        # check if output files are valid
        if len(output_files) != 6:
            raise ValueError("Output files must be a list of six file paths.")
        if not output_files[0].endswith(".txt"):
            raise ValueError("Output file #1 (classification report) must be a .txt file.")
        if not isinstance(output_files[1], tuple):
            raise ValueError("Output file #2 (confusion matrix title and plot) must be a tuple.")
        if not output_files[1][1].endswith(".png"):
            raise ValueError("Output file #2 (confusion matrix plot) must be a .png file.")
        if not output_files[2].endswith(".json"):
            raise ValueError("Output file #3 (matthews correlation coefficient) must be a .json file.")
        if not output_files[3].endswith(".json"):
            raise ValueError("Output file #4 (ROUGE scores) must be a .json file.")
        if not output_files[4].endswith(".json"):
            raise ValueError("Output file #5 (BLEU scores) must be a .json file.")
        if not output_files[5].endswith(".json"):
            raise ValueError("Output file #6 (COMET scores) must be a .json file.")

        # evaluate classifications
        classification_output_files = output_files[:3]
        Model_Evaluator._evaluate_classifications(AITA_classes, correct_AITA_classes, classification_type, classification_output_files)

        # evaluate justifications
        justification_output_files = output_files[3:]
        Model_Evaluator._evaluate_justifications(submission_texts, predictions, references, justification_output_files)

    def _evaluate_classifications(AITA_classes, correct_AITA_classes, classification_type, output_files):
        '''
        Evaluate the AITA classifications.

        Args:
            AITA_classes (list): A list of predicted AITA classes.
            correct_AITA_classes (list): A list of correct AITA classes.
            classification_type (str): The type of classification to evaluate - either multi or binary
            output_files (list): A list of file paths to write the results to.
                - 0 - string: classification report file (.txt)
                - 1 - tuple: confusion matrix plot tile and file (.png)
                - 2 - string: matthews correlation coefficient file (.json)

        Returns:
            None - Writes results to output files.
        '''

        # track samples with no class to mention in results
        no_class_counter = 0

        # get y_true and y_pred
        y_true, y_pred = [], []
        for l1, l2 in zip(AITA_classes, correct_AITA_classes):
            if l1 != "NO CLASS":
                y_pred.append(l1)
                y_true.append(l2)
            else:
                no_class_counter += 1
        print('Predictions with no AITA class:', no_class_counter)

        # get class names for multi or binary classification
        if classification_type == 'multi':
            class_names = ['NTA', 'NAH', 'ESH', 'INFO', 'YTA']
        elif classification_type == 'binary':
            class_names = ['NTA', 'YTA']
        else:
            raise ValueError("Classification type must be either 'multi' or 'binary'")
        
        # get classification stats report and save it to provided output
        classification_metrics = classification_report(y_true, y_pred, labels=class_names)
        with open(output_files[0], 'w') as f:
            f.write(classification_metrics)
            print('Classification report written to', output_files[0])

        # get confusion matrix and save it to provided output
        cm = confusion_matrix(y_true, y_pred, labels=class_names)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{output_files[1][0]}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(f"{output_files[1][1]}")
        print('Confusion matrix plot written to', output_files[1][1])

        # get matthews correlation coefficient and save it to JSON
        matthews_metric = evaluate.load("matthews_correlation")
        mcc = matthews_metric.compute(references=[class_names.index(x) for x in y_true], predictions=[class_names.index(x) for x in y_pred])
        with open(output_files[2], 'w') as f:
            json.dump({'mcc': mcc}, f)
            print('Matthews correlation coefficient written to', output_files[2])

    def _evaluate_justifications(submission_texts, predictions, references, output_files):
        '''
        Evaluate the justification texts.

        Args:
            submission_texts (list): A list of submission texts. (needed for COMET)
            predictions (list): A list of prediction texts.
            references (list): A list of reference texts.
            output_files (list): A list of file paths to write the results to.\
                - 0 - string: ROUGE scores file (.json)
                - 1 - string: BLEU scores file (.json)
                - 2 - string: COMET scores file (.json)

        Returns:
            None - Writes results to output files.
        '''

        # get ROUGE scores and save them to provided output
        rouge_metric = evaluate.load("rouge")
        rouge = rouge_metric.compute(predictions=predictions, references=references)

        with open(output_files[0], 'w') as f:
            json.dump(rouge, f)

        # get BLEU scores and save them to provided output
        bleu_metric = evaluate.load("bleu")
        bleu = bleu_metric.compute(predictions=predictions, references=references)

        with open(output_files[1], 'w') as f:
            json.dump(bleu, f)
        
        # get COMET scores
        comet_metric = evaluate.load('comet') 
        comet_score = comet_metric.compute(predictions=predictions, references=references, sources=submission_texts)

        with open(output_files[2], 'w') as f:
            json.dump(comet_score, f)
