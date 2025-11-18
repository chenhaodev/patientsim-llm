"""
Doctor Agent - Conducts medical interviews with patients
"""

from typing import Dict, List
from llm_client import LLMClient


class DoctorAgent:
    """Simulates a doctor conducting a medical interview"""

    def __init__(self, model_id: str, llm_client: LLMClient, patient_chief_complaint: str):
        """
        Initialize doctor agent

        Args:
            model_id: LLM model to use (e.g., 'gpt-4.1-api')
            llm_client: Initialized LLMClient instance
            patient_chief_complaint: Patient's chief complaint to guide interview
        """
        self.model_id = model_id
        self.client = llm_client
        self.chief_complaint = patient_chief_complaint
        self.conversation_history = []

        # Build system prompt
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build doctor system prompt"""

        prompt = f"""You are an experienced emergency department physician conducting a patient interview. Your goal is to gather comprehensive medical information to make an accurate diagnosis.

## CHIEF COMPLAINT
The patient presents with: {self.chief_complaint}

## YOUR RESPONSIBILITIES

1. **Conduct a thorough history:**
   - History of Present Illness (HPI): Onset, location, duration, characteristics, aggravating/relieving factors, radiation, timing, severity
   - Review of Systems (ROS): Systematic review of relevant systems
   - Past Medical History (PMH): Previous conditions, surgeries, hospitalizations
   - Medications: Current medications, dosages, compliance
   - Allergies: Drug allergies and reactions
   - Social History: Tobacco, alcohol, drugs, occupation, living situation
   - Family History: Relevant family medical history

2. **Ask focused, clear questions:**
   - One question at a time
   - Use open-ended questions initially, then follow up with specific questions
   - Adapt your language to the patient's comprehension level
   - Be empathetic and professional

3. **Build rapport:**
   - Show empathy and concern
   - Acknowledge the patient's discomfort
   - Explain your reasoning when appropriate

4. **Work toward diagnosis:**
   - Gather enough information to form a differential diagnosis
   - Ask follow-up questions based on patient responses
   - Consider red flags and serious conditions

## INTERVIEW STRUCTURE

Start with:
1. Introduction and opening question about the chief complaint
2. Detailed history of present illness
3. Associated symptoms (review of systems)
4. Past medical history and medications
5. Social and family history
6. Summarize findings and explain next steps

## CONVERSATION STYLE

- Be professional yet warm
- Use clear, simple language
- Ask one question at a time
- Listen carefully to responses
- Follow up on important details
- Adapt to the patient's communication style

## IMPORTANT RULES

1. Stay in character as a physician
2. Do not make diagnoses out loud (think through differential internally)
3. Focus on gathering information through questions
4. Be realistic - you cannot perform physical exams in this text conversation
5. If the patient seems confused or has language difficulties, adjust your approach
6. Respond naturally - no stage directions or meta-commentary

## RESPONSE FORMAT

Respond ONLY with what the doctor would say. Keep responses concise and focused.
"""

        return prompt

    def start_interview(self) -> str:
        """
        Start the interview with opening statement

        Returns:
            Doctor's opening message
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Begin the interview. The patient has come to the ED with: {self.chief_complaint}"}
        ]

        response = self.client.generate(
            model_id=self.model_id,
            messages=messages
        )

        # Add to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })

        return response

    def respond(self, patient_message: str, turn_number: int, max_turns: int) -> str:
        """
        Generate doctor's response to patient

        Args:
            patient_message: What the patient said
            turn_number: Current turn number
            max_turns: Maximum turns allowed

        Returns:
            Doctor's response
        """
        # Add patient message to history
        self.conversation_history.append({
            "role": "user",
            "content": f"Patient: {patient_message}"
        })

        # Build context message
        if turn_number >= max_turns - 2:
            context = f"\n\n[You are near the end of the interview (turn {turn_number}/{max_turns}). Start summarizing and explaining next steps.]"
        else:
            context = ""

        # Build messages for LLM
        messages = [
            {"role": "system", "content": self.system_prompt + context},
            *self.conversation_history
        ]

        response = self.client.generate(
            model_id=self.model_id,
            messages=messages
        )

        # Add doctor response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })

        return response

    def should_end_interview(self, turn_number: int, max_turns: int) -> bool:
        """
        Determine if interview should end

        Args:
            turn_number: Current turn number
            max_turns: Maximum allowed turns

        Returns:
            True if interview should end
        """
        return turn_number >= max_turns

    def summarize_findings(self) -> str:
        """
        Generate summary of findings (for internal use/logging)

        Returns:
            Summary of key findings from interview
        """
        messages = [
            {"role": "system", "content": "You are a physician. Summarize the key findings from this patient interview in 2-3 sentences."},
            *self.conversation_history,
            {"role": "user", "content": "Provide a brief clinical summary of this case."}
        ]

        summary = self.client.generate(
            model_id=self.model_id,
            messages=messages,
            max_tokens=200
        )

        return summary

    def reset_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []

    def get_metadata(self) -> Dict:
        """Return doctor metadata for logging"""
        return {
            "doctor_engine_name": self.model_id
        }
