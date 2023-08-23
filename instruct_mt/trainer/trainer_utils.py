from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import os
import shutil


class RemoveDeepspeedCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )

        global_step_path = os.path.join(checkpoint_folder, f"global_step{state.global_step}")
        if os.path.exists(global_step_path):
            if args.local_rank == 0:
                shutil.rmtree(global_step_path)
        return control