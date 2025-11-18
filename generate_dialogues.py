"""
Main script to generate patient-doctor dialogues
"""

import json
import jsonlines
import os
import yaml
from typing import List, Dict, Optional
from pathlib import Path
import argparse
from tqdm import tqdm

from llm_client import LLMClient
from patient_agent import PatientAgent
from doctor_agent import DoctorAgent


class DialogueGenerator:
    """Orchestrates dialogue generation between doctor and patient"""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize generator with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.llm_client = LLMClient(config_path)
        self.max_turns = self.config['simulation']['max_turns']
        self.output_dir = Path(self.config['simulation']['output_dir'])

        # Load patient profiles
        profile_path = self.config['patient_profile_path']
        with open(profile_path, 'r') as f:
            self.patient_profiles = json.load(f)

        print(f"Loaded {len(self.patient_profiles)} patient profiles")
        print(f"Available models: {self.llm_client.get_available_models()}")

    def generate_single_dialogue(self,
                                  profile: Dict,
                                  doctor_model: str,
                                  patient_model: str) -> Dict:
        """
        Generate a single dialogue between doctor and patient

        Args:
            profile: Patient profile dict
            doctor_model: Model ID for doctor
            patient_model: Model ID for patient

        Returns:
            Dialogue data dict
        """
        # Initialize agents
        patient = PatientAgent(
            profile=profile,
            model_id=patient_model,
            llm_client=self.llm_client
        )

        doctor = DoctorAgent(
            model_id=doctor_model,
            llm_client=self.llm_client,
            patient_chief_complaint=profile.get('chiefcomplaint', 'Not specified')
        )

        # Start dialogue
        dialog_history = []

        # Doctor starts
        doctor_message = doctor.start_interview()
        dialog_history.append({
            "role": "Doctor",
            "content": doctor_message
        })

        # Conversation loop
        for turn in range(self.max_turns):
            # Patient responds
            patient_message = patient.respond(doctor_message)
            dialog_history.append({
                "role": "Patient",
                "content": patient_message
            })

            # Check if should end
            if doctor.should_end_interview(turn + 1, self.max_turns):
                break

            # Doctor responds
            doctor_message = doctor.respond(patient_message, turn + 1, self.max_turns)
            dialog_history.append({
                "role": "Doctor",
                "content": doctor_message
            })

        # Build output
        dialogue_data = {
            **patient.get_metadata(),
            **doctor.get_metadata(),
            "dialog_history": dialog_history,
            "diagnosis": profile.get('diagnosis')
        }

        return dialogue_data

    def generate_for_split(self,
                           split: str,
                           doctor_model: str,
                           patient_model: str,
                           limit: Optional[int] = None) -> List[Dict]:
        """
        Generate dialogues for a specific data split

        Args:
            split: 'persona', 'info', or 'valid'
            doctor_model: Model ID for doctor
            patient_model: Model ID for patient
            limit: Optional limit on number of profiles to process

        Returns:
            List of dialogue dicts
        """
        # Filter profiles by split
        split_profiles = [p for p in self.patient_profiles if p.get('split') == split]

        if limit:
            split_profiles = split_profiles[:limit]

        print(f"\nGenerating {len(split_profiles)} dialogues for {split} split")
        print(f"Doctor: {doctor_model}, Patient: {patient_model}")

        dialogues = []
        for profile in tqdm(split_profiles, desc=f"Generating {split} dialogues"):
            try:
                dialogue = self.generate_single_dialogue(
                    profile=profile,
                    doctor_model=doctor_model,
                    patient_model=patient_model
                )
                dialogues.append(dialogue)

            except Exception as e:
                print(f"\nError processing hadm_id {profile.get('hadm_id')}: {str(e)}")
                continue

        return dialogues

    def save_dialogues(self, dialogues: List[Dict], output_path: Path):
        """Save dialogues to JSONL file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with jsonlines.open(output_path, 'w') as writer:
            for dialogue in dialogues:
                writer.write(dialogue)

        print(f"Saved {len(dialogues)} dialogues to {output_path}")

    def run_full_simulation(self,
                           doctor_model: str,
                           patient_model: str,
                           splits: List[str] = ['persona', 'info'],
                           limit: Optional[int] = None):
        """
        Run full simulation for specified splits

        Args:
            doctor_model: Model ID for doctor
            patient_model: Model ID for patient
            splits: List of splits to process
            limit: Optional limit per split
        """
        for split in splits:
            print(f"\n{'='*60}")
            print(f"Processing {split.upper()} split")
            print(f"{'='*60}")

            # Generate dialogues
            dialogues = self.generate_for_split(
                split=split,
                doctor_model=doctor_model,
                patient_model=patient_model,
                limit=limit
            )

            # Save to appropriate directory
            split_dir = self.output_dir / f"{split}_test" / "llm_simulation" / patient_model
            output_file = split_dir / "llm_dialogue.jsonl"

            self.save_dialogues(dialogues, output_file)

    def run_multi_model_simulation(self,
                                   doctor_model: str,
                                   patient_models: List[str],
                                   splits: List[str] = ['persona', 'info'],
                                   limit: Optional[int] = None):
        """
        Run simulation with one doctor model and multiple patient models

        Args:
            doctor_model: Model ID for doctor
            patient_models: List of patient model IDs
            splits: List of splits to process
            limit: Optional limit per split
        """
        for patient_model in patient_models:
            print(f"\n{'#'*60}")
            print(f"PATIENT MODEL: {patient_model}")
            print(f"{'#'*60}")

            self.run_full_simulation(
                doctor_model=doctor_model,
                patient_model=patient_model,
                splits=splits,
                limit=limit
            )


def main():
    parser = argparse.ArgumentParser(description='Generate patient-doctor dialogues')

    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--doctor-model', default='gpt-4.1-api', help='Doctor model ID')
    parser.add_argument('--patient-model', default='deepseek-api', help='Patient model ID (or comma-separated list)')
    parser.add_argument('--splits', default='persona,info', help='Comma-separated splits to process')
    parser.add_argument('--limit', type=int, help='Limit number of profiles per split (for testing)')
    parser.add_argument('--test-connection', action='store_true', help='Test API connections and exit')

    args = parser.parse_args()

    # Initialize generator
    generator = DialogueGenerator(config_path=args.config)

    # Test connections if requested
    if args.test_connection:
        print("\nTesting API connections...")
        for model_id in generator.llm_client.get_available_models():
            if generator.llm_client.test_connection(model_id):
                print(f"✓ {model_id}")
            else:
                print(f"✗ {model_id}")
        return

    # Parse patient models
    patient_models = [m.strip() for m in args.patient_model.split(',')]

    # Parse splits
    splits = [s.strip() for s in args.splits.split(',')]

    # Run simulation
    if len(patient_models) == 1:
        generator.run_full_simulation(
            doctor_model=args.doctor_model,
            patient_model=patient_models[0],
            splits=splits,
            limit=args.limit
        )
    else:
        generator.run_multi_model_simulation(
            doctor_model=args.doctor_model,
            patient_models=patient_models,
            splits=splits,
            limit=args.limit
        )

    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
