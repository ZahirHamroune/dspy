from .predict import Predict
from typing import  List
import dspy
import dsp


class SelfDiscoverySignature(dspy.Signature):

    task_to_resolve = dspy.InputField()
    # reasoning_modules = dspy.InputField()

    # selected_modules = dspy.OutputField(desc='SELECT relevant reasoning modules for the task') #internal state
    # adapted_modules = dspy.OutputField(desc='ADAPT the selected reasoning modules to be more specific to the task')  #internal state
    # implement_reasoning_structure = dspy.OutputField(desc='IMPLEMENT the adapted reasoning modules into an actionable reasoning structure')  #internal state
    # execute_reasoning_structure = dspy.OutputField(desc='the reasoning structure to solve a specific task instance')  #internal state

    anwser = dspy.OutputField(desc='Anwser')  #internal state

class SelfDiscovery(Predict):

    # STAGE 1

    def select_reasoning_modules_template(self, task_to_resolve, reasoning_modules):
        """
        Step 1: SELECT relevant reasoning modules for the task.
        """
        return dsp.Type(
            prefix=f"which of the following reasoning modules are relevant? Do not elaborate on why: {reasoning_modules}",
            desc="SELECT relevant reasoning modules for the task."
        )

    def adapt_reasoning_modules_template(self, task_to_resolve, selected_modules):
        """
        Step 2: ADAPT the selected reasoning modules to be more specific to the task.
        """
        return dsp.Type(
            prefix=f"Without working out the full solution, adapt the following reasoning modules to be specific to our task:\n{selected_modules}",
            desc=f"ADAPT the selected reasoning modules to be more specific to the task"
        )

    def implement_reasoning_structure_template(self, task_to_resolve, adapted_modules ):
        """
        Step 3: IMPLEMENT the adapted reasoning modules into an actionable reasoning structure.
        """
        # prompt = f"Without working out the full solution, create an actionable reasoning structure for the task using these adapted reasoning modules:\n{adapted_modules}\n\nTask Description:\n{task_to_resolve}"
        return dsp.Type(
            prefix=f"Without working out the full solution, create an actionable reasoning structure for the task using these adapted reasoning modules:\n{adapted_modules}",
            desc=f"IMPLEMENT the adapted reasoning modules into an actionable reasoning structure"
        )

    # STAGE 2

    def execute_reasoning_structure_template(self, task_to_resolve, reasoning_structure):
        """
        Execute the reasoning structure to solve a specific task instance.
        """
        # prompt = f"Using the following reasoning structure: {reasoning_structure}\n\nSolve this task, providing your final answer: {task_to_resolve}"

        return dsp.Type(
            prefix=f"Using the following reasoning structure: {reasoning_structure}, Solve this task, providing your final answer: {task_to_resolve}",
            desc=f"Execute the reasoning structure to solve a specific task instance"
        )



    def __init__(self, signature:SelfDiscoverySignature, **config):
        super().__init__(signature, **config)

        self.signature = signature


    def forward(self, **kwargs):
        task_to_resolve = kwargs.get("task_to_resolve")
        reasoning_modules = kwargs.get("reasoning_modules")
        signature = self.signature
        *keys, last_key = signature.kwargs.keys()

        extended_kwargs = {key: signature.kwargs[key] for key in keys}
        extended_kwargs.update(
            {"select_reasoning_modules": self.select_reasoning_modules_template(task_to_resolve, reasoning_modules), last_key: signature.kwargs[last_key]}
        )

        #SELECT
        self.selection_signature = dsp.Template(
            signature.instructions, **extended_kwargs
        )
        selection_modules = super().forward(signature=self.selection_signature, **kwargs).anwser
        print("selection_modules ==>", selection_modules)

        #ADAPT
        extended_kwargs = {key: signature.kwargs[key] for key in keys}
        extended_kwargs.update(
            {"adapt_reasoning_modules": self.adapt_reasoning_modules_template(task_to_resolve, selection_modules), last_key: signature.kwargs[last_key]}
        )
        self.adapt_signature = dsp.Template(
            signature.instructions, **extended_kwargs
        )

        adapt_result = super().forward(signature=self.adapt_signature, **kwargs).anwser
        print("Adapt ==>", adapt_result)

        # IMPLEMENT
        extended_kwargs = {key: signature.kwargs[key] for key in keys}
        extended_kwargs.update(
            {"implement_reasoning_structure": self.implement_reasoning_structure_template(task_to_resolve, adapt_result), last_key: signature.kwargs[last_key]}
        )
        self.implement_signature = dsp.Template(
            signature.instructions, **extended_kwargs
        )

        implement_result = super().forward(signature=self.implement_signature, **kwargs).anwser
        print("implement_result ==>", implement_result)


        #SOLVE
        extended_kwargs = {key: signature.kwargs[key] for key in keys}
        extended_kwargs.update(
            {"execute_reasoning_structure": self.execute_reasoning_structure_template(task_to_resolve, implement_result),
             last_key: signature.kwargs[last_key]}
        )
        self.execute_template = dsp.Template(
            signature.instructions, **extended_kwargs
        )

        final_solution = super().forward(signature=self.execute_template, **kwargs)
        execute_result = final_solution.anwser
        print("execute_template ==>", execute_result)

        return final_solution