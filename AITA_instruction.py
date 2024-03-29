import re

class AITA_Instruction:
    def __init__(self):
        pass

    AITA_classifications = [
        "you're the a**hole (yta) when the writer is causing the conflict",
        "not the a**hole (nta) when someone other than the writer is causing the conflict",
        "no a**holes here (nah) when no one is causing the conflict",
        "everyone sucks here (esh) when everyone is causing the conflict",
        "more information needed (info) when a classification cannot be made with the context provided",
    ]
    MULTICLASS_INSTRUCTION_PREFIX = f"Classify the interpersonal conflict and explain your choice. Your choices are limited to {', '.join(AITA_classifications)}."
    BINARY_INSTRUCTION_PREFIX = f"Classify the interpersonal conflict as either 'you're the a**hole (yta)' when the writer is causing the conflict or 'not the a**hole (nta)' when someone other than the writer is causing the conflict and explain your choice."

    @staticmethod
    def get_flanT5_instruction(sample, instruction_type):
        submission_text = re.sub(r' +', ' ', re.sub(r'\n+', ' ', sample["submission_text"]))
        if instruction_type == "binary":
            flanT5_instruction = AITA_Instruction.BINARY_INSTRUCTION_PREFIX + " Conflict context: " + submission_text
        elif instruction_type == "multiclass":
            flanT5_instruction = AITA_Instruction.MULTICLASS_INSTRUCTION_PREFIX + " Conflict context: " + submission_text
        else:
            raise ValueError("Instruction type must be either 'binary' or 'multiclass'")
        return {"flanT5_instruction": flanT5_instruction}
    
    @staticmethod
    def get_llama2_training_instruction(sample, instruction_type, partition):
        submission_text = re.sub(r' +', ' ', re.sub(r'\n+', ' ', sample["submission_text"]))
        top_comment_text = re.sub(r' +', ' ',  re.sub(r'\n+', ' ', sample["top_comment_1"]))

        if instruction_type == "binary":
            if partition == "training":
                llama2_instruction = f"<s>[INST] <<SYS>>\n{AITA_Instruction.BINARY_INSTRUCTION_PREFIX}\n<</SYS>>\n\n{submission_text} [/INST] {top_comment_text} </s>"
            elif partition == "testing":
                llama2_instruction = f"<s>[INST] <<SYS>>\n{AITA_Instruction.BINARY_INSTRUCTION_PREFIX}\n<</SYS>>\n\n{submission_text} [/INST]"
            else:
                raise ValueError("Partition must be either 'training' or 'testing'")
        elif instruction_type == "multiclass":
            if partition == "training":
                llama2_instruction = f"<s>[INST] <<SYS>>\n{AITA_Instruction.MULTICLASS_INSTRUCTION_PREFIX}\n<</SYS>>\n\n{submission_text} [/INST] {top_comment_text} </s>"
            elif partition == "testing":
                llama2_instruction = f"<s>[INST] <<SYS>>\n{AITA_Instruction.MULTICLASS_INSTRUCTION_PREFIX}\n<</SYS>>\n\n{submission_text} [/INST]"
            else:
                raise ValueError("Partition must be either 'training' or 'testing'")
        else:
            raise ValueError("Instruction type must be either 'binary' or 'multiclass'")

        return {"llama2_instruction": llama2_instruction}
