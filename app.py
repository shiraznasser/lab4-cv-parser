import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
import torch
import re
import json
import tempfile

st.set_page_config(page_title="HR Candidate Profile Parser", layout="centered")
st.title("HR Candidate Profile Parser")
st.write("Upload a CV as a PDF and extract a clean JSON candidate profile.")

MODEL_NAME = "mistralai/Mistral-Nemo-Instruct-2407"


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    return tokenizer, model


def generate_text(tokenizer, model, prompt, max_length=5000, num_return_sequences=1):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
    )
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


def extract_json_block(text):
    pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    return f"```json\n{matches[-1]}\n```"


full_name_schema = ResponseSchema(
    name="full_name",
    description="The full name of the candidate."
)
email_schema = ResponseSchema(
    name="email",
    description="The email address of the candidate."
)
education_schema = ResponseSchema(
    name="education",
    description="A list of objects, each with the keys degree (string), institution (string) and year (integer)."
)
skills_schema = ResponseSchema(
    name="skills",
    description="A list of skills, each one as a string."
)
experience_schema = ResponseSchema(
    name="experience",
    description="A list of objects, each with the keys role (string), company (string) and years (string, for example 2020-2023)."
)

response_schemas = [full_name_schema, email_schema, education_schema, skills_schema, experience_schema]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

cv_extraction_template = """
You are a smart HR assistant that extracts a candidate profile from a resume.

Extract the full name, email, education, skills and experience of the candidate.

Respond ONLY in JSON format as follows:
{format_instructions}

Now extract from the following resume:
"{cv_text}"
"""

uploaded_file = st.file_uploader("Upload CV (PDF)", type=["pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    cv_text = "\n".join([doc.page_content for doc in documents])



    if st.button("Parse CV"):
        with st.spinner("Loading model and generating profile..."):
            tokenizer, model = load_model()

            prompt = PromptTemplate(
                template=cv_extraction_template,
                input_variables=["cv_text", "format_instructions"]
            ).format(cv_text=cv_text, format_instructions=format_instructions)

            response = generate_text(tokenizer, model, prompt)[0]

            try:
                json_text = extract_json_block(response)
                output_data = output_parser.parse(json_text)
                st.subheader("Candidate Profile (JSON)")
                st.json(output_data)
            except Exception as e:
                st.error(f"Could not parse the model output: {e}")
                st.subheader("Raw Model Output")
                st.text(response)
else:
    st.info("Upload a CV PDF to get started.")
