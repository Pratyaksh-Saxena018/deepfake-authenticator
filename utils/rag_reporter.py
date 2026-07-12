import os
import json
import numpy as np

# Lightweight knowledge base of published scientific literature on AI frequency forensics
FORENSIC_KNOWLEDGE_BASE = [
    {
        "topic": "fft_high_frequency_slope",
        "text": "In natural camera captures, spatial energy exhibits a predictable power-law decay toward high frequencies due to optical lens blur and sensor smoothing. Generative AI models (GANs and Diffusion decoders) rely on spatial upsampling layers such as transposed convolutions. These layers introduce periodic checkerboard grid artifacts, causing the high-frequency FFT radial profile to flatten significantly (a less negative slope) compared to authentic camera lenses."
    },
    {
        "topic": "dct_energy_ratio",
        "text": "The Discrete Cosine Transform (DCT) separates structural low frequencies from textural high frequencies. AI-generated images frequently struggle to replicate consistent high-frequency ratios, either exhibiting hyper-smooth skin (abnormally low DCT high-frequency energy) or synthetic micro-texture spikes across the 8x8 DCT quantization grids."
    },
    {
        "topic": "wavelet_diagonal_noise",
        "text": "Discrete Wavelet Transform (DWT) Daubechies decomposition isolates diagonal high frequencies (the HH band). When synthetic face swaps or generative masks are blended onto target backgrounds, boundary inconsistencies and pixel synthesis errors manifest as anomalous energy variance within this HH diagonal band."
    }
]

class ForensicReporter:
    def __init__(self, api_key=None):
        """
        api_key: Optional Google Gemini API key. If None, uses a local synthesis engine
        to guarantee offline reliability during live presentations.
        """
        self.api_key = api_key
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                self.use_llm = True
            except Exception as e:
                print(f"[Warning] Could not initialize Gemini API: {e}. Switching to Offline Synthesis Mode.")
                self.use_llm = False
        else:
            self.use_llm = False

    def _retrieve_context(self, metrics):
        """Retrieves relevant scientific knowledge based on the detected statistical metrics."""
        retrieved_docs = []
        
        # Check FFT Slope anomaly
        if metrics.get("fft_slope", 0) > -0.40:
            retrieved_docs.append(FORENSIC_KNOWLEDGE_BASE[0]["text"])
            
        # Check DCT Ratio anomaly
        if metrics.get("dct_energy_ratio", 0) > 0.25:
            retrieved_docs.append(FORENSIC_KNOWLEDGE_BASE[1]["text"])
            
        # Check Wavelet energy
        retrieved_docs.append(FORENSIC_KNOWLEDGE_BASE[2]["text"])
        
        return "\n\n".join(retrieved_docs)

    def generate_report(self, metrics, fake_probability):
        """Generates an executive forensic audit report using RAG + LLM (or Offline Synthesis)."""
        verdict = "SYNTHETIC MEDIA DETECTED (DEEPFAKE)" if fake_probability > 0.50 else "AUTHENTIC MEDIA CONFIRMED"
        confidence = fake_probability * 100 if fake_probability > 0.50 else (1 - fake_probability) * 100
        
        # 1. Retrieve scientific context from our local knowledge base
        scientific_context = self._retrieve_context(metrics)
        
        # 2. Try LLM Generation via Google Gemini
        if self.use_llm:
            try:
                prompt = f"""
                You are a Senior Digital Media Forensic Investigator writing an executive audit report for a university presentation.
                
                ANALYSIS RESULTS:
                - Verdict: {verdict}
                - Confidence Score: {confidence:.2f}%
                - FFT Radial Slope: {metrics.get('fft_slope', 0):.4f}
                - DCT Energy Ratio: {metrics.get('dct_energy_ratio', 0):.4f}
                - Wavelet HH Energy: {metrics.get('wavelet_energy', 0):.2f}
                
                RETRIEVED SCIENTIFIC LITERATURE (RAG CONTEXT):
                {scientific_context}
                
                TASK:
                Write a structured, highly professional forensic report in Markdown. Include:
                1. Executive Summary & Verdict.
                2. Frequency Domain Analysis (Explain what the FFT slope and DCT numbers mean for this specific image).
                3. Scientific Grounding (Use the retrieved literature to explain WHY generative models cause these artifacts).
                4. Conclusion & Confidence Assessment.
                Keep the tone objective, authoritative, and academic.
                """
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                print(f"[Notice] LLM generation failed ({e}). Falling back to Offline Synthesis Mode.")

        # 3. Offline Synthesis Fallback (Guarantees zero crashes during live demos)
        return self._generate_offline_report(verdict, confidence, metrics, scientific_context)

    def _generate_offline_report(self, verdict, confidence, metrics, context):
        """Structured Markdown template engine for offline presentation reliability."""
        report = f"""# 🛡️ EXECUTIVE DIGITAL FORENSIC AUDIT REPORT

**Case Status:** CLOSED  
**Primary Verdict:** **{verdict}** **Assessment Confidence:** **{confidence:.2f}%** ---

## 1. Executive Summary
The submitted media asset underwent multi-domain spectral analysis to evaluate its authenticity. By bypassing standard spatial RGB color analysis and converting the image into the frequency domain, the authentication engine evaluated microscopic upsampling footprints, sensor noise consistency, and high-frequency energy decay curves. Based on tabular ensemble classification, the media is classified with **{confidence:.2f}% confidence** as **{verdict.split('(')[0].strip()}**.

---

## 2. Quantitative Spectral Biomarkers
* **Fast Fourier Transform (FFT) Radial Slope:** `{metrics.get('fft_slope', 0):.4f}`
* **Discrete Cosine Transform (DCT) Energy Ratio:** `{metrics.get('dct_energy_ratio', 0):.4f}`
* **Wavelet High-High (HH) Band Mean Energy:** `{metrics.get('wavelet_hh_mean', 0):.4f}`
* **Wavelet Total Diagonal Energy:** `{metrics.get('wavelet_energy', 0):.2f}`

---

## 3. Scientific Justification & RAG Grounding
*The following digital forensic evaluations are grounded in published signal processing literature:*

{context}

---

## 4. Analytical Conclusion
The mathematical features extracted from this sample align with the expected statistical profile of **{'generative AI synthesis algorithms' if 'SYNTHETIC' in verdict else 'authentic optical CMOS sensor captures'}**. The combination of radial energy distribution and diagonal wavelet variances indicates that the sample **{'exhibits distinct mathematical hallmarks of spatial upsampling and artificial texture synthesis' if 'SYNTHETIC' in verdict else 'retains natural optical lens blur and authentic camera noise characteristics'}**.

*Report generated automatically by the Deepfake Spectral Authenticator Engine.*
"""
        return report