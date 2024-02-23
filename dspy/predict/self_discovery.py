import dsp
from .predict import Predict
from typing import List
import dspy


class SelfDiscoverySignature(dspy.Signature):
    task_to_resolve = dspy.InputField()
    # reasoning_modules = dspy.InputField()

    # selected_modules = dspy.OutputField(desc='SELECT relevant reasoning modules for the task') #internal state
    # adapted_modules = dspy.OutputField(desc='ADAPT the selected reasoning modules to be more specific to the task')  #internal state
    # implement_reasoning_structure = dspy.OutputField(desc='IMPLEMENT the adapted reasoning modules into an actionable reasoning structure')  #internal state
    # execute_reasoning_structure = dspy.OutputField(desc='the reasoning structure to solve a specific task instance')  #internal state

    anwser = dspy.OutputField(desc='Anwser')  # internal state


class SelfDiscovery(Predict):

    # STAGE 1

    def _select_reasoning_modules_template(self, reasoning_modules):
        """
        Step 1: SELECT relevant reasoning modules for the task.
        """
        return dsp.Type(
            prefix=f"which of the following reasoning modules are relevant? Do not elaborate on why: {reasoning_modules}",
            desc="SELECT relevant reasoning modules for the task."
        )

    def _adapt_reasoning_modules_template(self, selected_modules):
        """
        Step 2: ADAPT the selected reasoning modules to be more specific to the task.
        """
        return dsp.Type(
            prefix=f"Without working out the full solution, adapt the following reasoning modules to be specific to our task:\n{selected_modules}",
            desc=f"ADAPT the selected reasoning modules to be more specific to the task"
        )

    def _implement_reasoning_structure_template(self, adapted_modules):
        """
        Step 3: IMPLEMENT the adapted reasoning modules into an actionable reasoning structure.
        """
        # prompt = f"Without working out the full solution, create an actionable reasoning structure for the task using these adapted reasoning modules:\n{adapted_modules}\n\nTask Description:\n{task_to_resolve}"
        return dsp.Type(
            prefix=f"Without working out the full solution, create an actionable reasoning structure for the task using these adapted reasoning modules:\n{adapted_modules}",
            desc=f"IMPLEMENT the adapted reasoning modules into an actionable reasoning structure"
        )

    # STAGE 2

    def _execute_reasoning_structure_template(self, task_to_resolve, reasoning_structure):
        """
        Execute the reasoning structure to solve a specific task instance.
        """
        # prompt = f"Using the following reasoning structure: {reasoning_structure}\n\nSolve this task, providing your final answer: {task_to_resolve}"

        return dsp.Type(
            prefix=f"Using the following reasoning structure: {reasoning_structure}, Solve this task, providing your final answer: {task_to_resolve}",
            desc=f"Execute the reasoning structure to solve a specific task instance"
        )

    def __init__(self, signature: SelfDiscoverySignature, **config):
        super().__init__(signature, **config)
        self.signature = signature

    def forward(self, **kwargs):
        task_to_resolve = kwargs.get("task_to_resolve")
        reasoning_modules: List[str] = kwargs.get("reasoning_modules")

        selection_modules = self.selectStep(reasoning_modules, task_to_resolve)

        adapt_result = self.adaptStep(selection_modules, task_to_resolve)

        implement_result = self.implementStep(adapt_result, task_to_resolve)

        solution = self.solveStep(implement_result, task_to_resolve)

        return solution

    def selectStep(self, reasoning_modules: List[str], task_to_resolve: str):
        kwargs = {"task_to_resolve": task_to_resolve, "reasoning_modules": "\n".join(reasoning_modules)}
        signature = self.signature
        *keys, output_key = signature.kwargs.keys()
        extended_kwargs = {key: signature.kwargs[key] for key in keys}
        extended_kwargs.update(
            {"select_reasoning_modules": self._select_reasoning_modules_template(reasoning_modules),
             output_key: signature.kwargs[output_key]}
        )
        # SELECT
        selection_signature = dsp.Template(
            signature.instructions, **extended_kwargs
        )
        selection_modules = super().forward(signature=selection_signature, **kwargs).anwser
        return selection_modules

    def adaptStep(self, selection_modules: str, task_to_resolve: str):
        kwargs = {"task_to_resolve": task_to_resolve, "selection_modules": selection_modules}
        signature = self.signature
        *keys, output_key = signature.kwargs.keys()
        extended_kwargs = {key: signature.kwargs[key] for key in keys}
        extended_kwargs.update(
            {"adapt_reasoning_modules": self._adapt_reasoning_modules_template(selection_modules),
             output_key: signature.kwargs[output_key]}
        )
        adapt_signature = dsp.Template(
            signature.instructions, **extended_kwargs
        )
        adapt_result = super().forward(signature=adapt_signature, **kwargs).anwser
        return adapt_result

    def implementStep(self, adapt_result: str, task_to_resolve: str):
        kwargs = {"task_to_resolve": task_to_resolve, "adapt_result": adapt_result}
        signature = self.signature
        *keys, output_key = signature.kwargs.keys()
        extended_kwargs = {key: signature.kwargs[key] for key in keys}
        extended_kwargs.update(
            {"implement_reasoning_structure": self._implement_reasoning_structure_template(adapt_result),
             output_key: signature.kwargs[output_key]}
        )
        implement_signature = dsp.Template(
            signature.instructions, **extended_kwargs
        )
        implement_result = super().forward(signature=implement_signature, **kwargs).anwser
        return implement_result

    def solveStep(self, implement_result: str, task_to_resolve: str):
        kwargs = {"task_to_resolve": task_to_resolve, "implement_result": implement_result}
        signature = self.signature
        *keys, output_key = signature.kwargs.keys()
        extended_kwargs = {key: signature.kwargs[key] for key in keys}
        extended_kwargs.update(
            {"execute_reasoning_structure": self._execute_reasoning_structure_template(task_to_resolve,implement_result),
             output_key: signature.kwargs[output_key]}
        )
        execute_template = dsp.Template(
            signature.instructions, **extended_kwargs
        )
        solution = super().forward(signature=execute_template, **kwargs)

        return solution
