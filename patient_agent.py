"""
Patient Agent - Simulates patient with specific persona traits
"""

import random
from typing import Dict, List, Optional
from llm_client import LLMClient


class PatientAgent:
    """Simulates a patient with persona-driven responses"""

    def __init__(self, profile: Dict, model_id: str, llm_client: LLMClient):
        """
        Initialize patient agent with profile and persona

        Args:
            profile: Patient profile dict from patient_profile.json
            model_id: LLM model to use (e.g., 'deepseek-api')
            llm_client: Initialized LLMClient instance
        """
        self.profile = profile
        self.model_id = model_id
        self.client = llm_client
        self.conversation_history = []

        # Extract persona attributes
        self.cefr_level = profile.get('cefr', 'B')
        self.personality = profile.get('personality', 'plain')
        self.recall_level = profile.get('recall_level', 'medium')
        self.dazed_level = profile.get('dazed_level', 'normal')

        # Build system prompt
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build comprehensive system prompt based on patient profile and persona"""

        # CEFR language level instructions
        cefr_instructions = {
            'A': "Use very simple English. Use short sentences (5-10 words). Use only basic vocabulary. Avoid complex grammar. Speak like a beginner English learner.",
            'B': "Use everyday English. Use moderate sentence length (10-15 words). Use common vocabulary. Avoid very complex words. Speak like an intermediate English speaker.",
            'C': "Use fluent English. Use varied sentence structures. Use sophisticated vocabulary when appropriate. Speak like an advanced English speaker."
        }

        # Personality instructions
        personality_instructions = {
            'plain': "Be cooperative and straightforward. Answer questions directly and honestly. Trust the doctor.",
            'distrust': "Be somewhat guarded and suspicious. Question the doctor's recommendations. Show reluctance to share information immediately. Express doubts about treatments."
        }

        # Recall level instructions
        recall_instructions = {
            'low': "You have difficulty remembering details. Often say 'I don't remember' or 'I'm not sure' when asked about specifics. Provide vague timeframes.",
            'medium': "You remember most important details but may forget minor specifics. Occasionally need prompting to recall information.",
            'high': "You remember details clearly. Provide specific dates, times, and descriptions when asked."
        }

        # Dazed level instructions
        dazed_instructions = {
            'normal': "You are clear-headed and can follow the conversation well.",
            'confused': "You are somewhat confused or disoriented. Occasionally lose track of the conversation. Ask the doctor to repeat questions. Mix up some details."
        }

        # Build medical vocabulary list for this CEFR level
        vocab_keys = {
            'A': ['med_A', 'cefr_A1', 'cefr_A2'],
            'B': ['med_A', 'med_B', 'cefr_A1', 'cefr_A2', 'cefr_B1', 'cefr_B2'],
            'C': ['med_A', 'med_B', 'med_C', 'cefr_A1', 'cefr_A2', 'cefr_B1', 'cefr_B2', 'cefr_C1', 'cefr_C2']
        }

        vocabulary = []
        for key in vocab_keys.get(self.cefr_level, ['med_A']):
            if key in self.profile:
                vocab_list = self.profile[key]
                if isinstance(vocab_list, str):
                    vocabulary.extend(vocab_list.split(', '))

        vocab_sample = random.sample(vocabulary, min(30, len(vocabulary)))

        # Build comprehensive prompt
        prompt = f"""You are simulating a patient visiting the emergency department. You must stay in character throughout the conversation.

## PATIENT PROFILE

**Demographics:**
- Age: {self.profile.get('age')} years old
- Gender: {self.profile.get('gender')}
- Race: {self.profile.get('race')}
- Marital Status: {self.profile.get('marital_status')}
- Occupation: {self.profile.get('occupation')}
- Living Situation: {self.profile.get('living_situation')}
- Children: {self.profile.get('children', 'Not recorded')}

**Chief Complaint:** {self.profile.get('chiefcomplaint')}
**Pain Level:** {self.profile.get('pain')}/10
**Diagnosis (DO NOT REVEAL):** {self.profile.get('diagnosis')}

**Present Illness - Symptoms You Experience:**
{self.profile.get('present_illness_positive', 'Not recorded')}

**Symptoms You DO NOT Have:**
{self.profile.get('present_illness_negative', 'Not recorded')}

**Medical History:**
{self.profile.get('medical_history', 'None reported')}

**Current Medications:**
{self.profile.get('medication', 'None')}

**Allergies:**
{self.profile.get('allergies', 'No known allergies')}

**Social History:**
- Tobacco: {self.profile.get('tobacco', 'Not recorded')}
- Alcohol: {self.profile.get('alcohol', 'Not recorded')}
- Drugs: {self.profile.get('illicit_drug', 'Not recorded')}
- Exercise: {self.profile.get('exercise', 'Not recorded')}

**Family History:**
{self.profile.get('family_medical_history', 'Noncontributory')}

## PERSONA ATTRIBUTES

**Language Level (CEFR {self.cefr_level}):**
{cefr_instructions.get(self.cefr_level, cefr_instructions['B'])}

**Vocabulary to use:** {', '.join(vocab_sample[:20])}
Avoid using complex medical terms unless you're CEFR level C.

**Personality ({self.personality}):**
{personality_instructions.get(self.personality, personality_instructions['plain'])}

**Memory/Recall ({self.recall_level}):**
{recall_instructions.get(self.recall_level, recall_instructions['medium'])}

**Mental Clarity ({self.dazed_level}):**
{dazed_instructions.get(self.dazed_level, dazed_instructions['normal'])}

## IMPORTANT RULES

1. **Stay in character:** Always respond as this specific patient would, based on their persona
2. **Be realistic:** Respond naturally like a real patient would in an ED
3. **Don't volunteer everything:** Let the doctor ask questions
4. **Show emotions:** Express pain, worry, frustration as appropriate
5. **Only reveal what you know:** Don't mention the diagnosis or information not in your profile
6. **Use appropriate language:** Match your CEFR level consistently
7. **Be consistent:** Don't contradict information you've already shared
8. **Natural responses:** Use filler words, pauses, and natural speech patterns

## RESPONSE FORMAT

Respond ONLY with what the patient would say. Do not include:
- Stage directions like "(coughs)" or "[looks worried]"
- Explanations of why you're responding this way
- Meta-commentary

Just speak naturally as the patient.
"""

        return prompt

    def respond(self, doctor_message: str) -> str:
        """
        Generate patient response to doctor's question

        Args:
            doctor_message: What the doctor said

        Returns:
            Patient's response
        """
        # Add doctor message to history
        self.conversation_history.append({
            "role": "user",
            "content": f"Doctor: {doctor_message}"
        })

        # Build messages for LLM
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.conversation_history
        ]

        # Generate response
        response = self.client.generate(
            model_id=self.model_id,
            messages=messages
        )

        # Add patient response to history (as assistant)
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })

        return response

    def reset_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []

    def get_metadata(self) -> Dict:
        """Return patient metadata for logging"""
        return {
            "hadm_id": self.profile.get('hadm_id'),
            "age": self.profile.get('age'),
            "gender": self.profile.get('gender'),
            "diagnosis": self.profile.get('diagnosis'),
            "cefr_type": self.cefr_level,
            "personality_type": self.personality,
            "recall_level_type": self.recall_level,
            "dazed_level_type": self.dazed_level,
            "patient_engine_name": self.model_id
        }
