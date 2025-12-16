import os
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import Model
from typing import Dict, TypedDict, Optional
from langgraph.graph import StateGraph, END
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

def initialize_and_run_workflow():
    # Install required packages
    os.system('pip install ibm-watsonx-ai')
    os.system('pip install -U langgraph')

    # Load environment variables
    load_dotenv()

    # Get credentials from environment variables
    def get_credentials():
        return {
            "url": os.getenv("IBM_CLOUD_URL"),
            "apikey": os.getenv("IBM_CLOUD_API_KEY")
        }

    # Model parameters and initialization
    # model_llama = "mistralai/mistral-large" #INIITAL MODEL USED
    model_llama = "ibm/granite-3-3-8b-instruct"
    # model_llama = "meta-llama/llama-3-3-70b-instruct"
    project_id = os.getenv("PROJECT_ID")
    parameters = {
        "decoding_method": "sample",
        "max_new_tokens": 10000,
        "random_seed": 3,
        "temperature": 0.35,
        "top_k": 50,
        "top_p": 1,
        "repetition_penalty": 1
    }

    print("Initializing LLM model...")
    model = Model(
        model_id=model_llama,
        params=parameters,
        credentials=get_credentials(),
        project_id=project_id
    )

    def llm(x):
        return model.generate_text(x)

    # Define the GraphState TypedDict
    class GraphState(TypedDict):
        feedback: Optional[str]
        history: Optional[str]
        code: Optional[str]
        specification: Optional[str]
        rating: Optional[str]
        iterations: Optional[int]
        updated_unit_test_cases: Optional[str]
        actual_code: Optional[str]
        sample_unit_test: Optional[str]

    # Initialize the StateGraph
    workflow = StateGraph(GraphState)

    # Define the prompts
    reviewer_start= """
    You are Junit test cases reviewer specialized in Junit test cases reviewing.
    You need to review the given code and Junit test and point out issues as bullet list.
    Code:\n {}
    Junit:\n {}
    """
    coder_start = """
    ROLE: You are an excellent agent focused on creating accurate and efficient Java code. 
        JOB: Consider the provided older version of pseudocode, older version of javacode as a example java code. Generate the New version Java code according to New version Pseudocode. New version Java code must contain the updated logic made in the New version Pseudocode. 
        RESTRICTIONS:
        1. Required to generate the code without object oriented programming concepts. Only code snippet is required.
        2. First Understand the older version pseudocode and its corresponding older version of javacode then compare the New version Pseudocode with older version Pseudocode. Strictly consider the updation made in New version Pseudocode.
        3. You must use the same parameter names for the java function call which was mention in the older version Java Code. 
        4. You may use the same variable names of older version Java Code in New version Java code.
        5. The changes made in the new version Java code must be included in the New version Java code.
        6. Use the English translated variable names from the Japanese written New version Pseudocode.
        7. Use the java built-in constant null value to validate the variables with null.
        8. Variable declaration and initialisation are required only when its mentioned in New version Pseudocode.

        Example older version Pseudocode:
        {}
        
        Example older version Java Code:
        {}
        
        Input New version Pseudocode: 
        {}
        
        
        FORMAT: 
        1. Strictly follow that you should print the new version java code inside this grave accent symbol.
        ```
        ```
        Output New version Java code:
        Output:
    """
    classify_feedback = "Are all feedback mentioned resolved in the code? Output just Yes or No.\
                        Code: \n {} \n Feedback: \n {} \n"

    # Define the handler functions
    def handle_reviewer(state):
        history = state.get('history', '').strip()
        code = state.get('code', '').strip()
        unit_test_cases = state.get('sample_unit_test', '').strip()
        print("Reviewer working...")
        feedback = llm(reviewer_start.format(code, unit_test_cases))
        return {'history': history + "\n REVIEWER:\n" + feedback, 'feedback': feedback, 'iterations': state.get('iterations') + 1}

    def handle_coder(state):
        history = state.get('history', '').strip()
        feedback = state.get('feedback', '').strip()
        code = state.get('code', '').strip()
        unit_test_cases = state.get('sample_unit_test', '').strip()
        print("CODER rewriting...")
        updated_code = llm(coder_start.format(feedback, code, unit_test_cases))
        return {'history': history + '\n CODER:\n' + updated_code, 'updated_unit_test_cases': updated_code}

    def deployment_ready(state):
        deployment_ready = 1 if 'yes' in llm(classify_feedback.format(state.get('CODER'), state.get('feedback'))) else 0
        total_iterations = 1 if state.get('iterations') > 5 else 0
        return "handle_result" if deployment_ready or total_iterations else "handle_coder"

    # Add nodes and edges to the workflow
    workflow.add_node("handle_reviewer", handle_reviewer)
    workflow.add_node("handle_coder", handle_coder)
    workflow.add_conditional_edges(
        "handle_reviewer",
        deployment_ready,
        {
            "handle_coder": "handle_coder",
            "handle_result": END,
        }
    )
    workflow.set_entry_point("handle_reviewer")
    workflow.add_edge("handle_reviewer", 'handle_coder')

    # Sample code and test case
    codes = """  
    public class Sample1{
        public int multiply(int x, int y){
            return x * y;
        }
        public int divide(int x, int y){
            return x / y;
        }
    }
    """

    sample_test = """   
    public class Sample1test{
        @Test
        public void multiplyTest1() {
            Sample1 sample1 = new  Sample1();
            int expected = 12;
            int actual = Sample1.multiply(3,4);
            assertThat(actual).isSameAs(expected);
        }
        @Test
        public void divideTest1() {
            Sample1 sample1 = new  Sample1();
            int expected = 3;
            int actual = Sample1.divide(12,4);
            assertThat(actual).isSameAs(expected);
        }
    }
    """
    app = workflow.compile()

    # Compile and run the workflow
    for output in app.stream({"history": codes, "code": codes, 'actual_code': codes, "sample_unit_test": sample_test, 'iterations': 0}, {"recursion_limit": 100}):
        for key, value in output.items():
            print(f"value from the node {key}")
            print("-----------")
            print(value['history'])
        print("\n=========\n")

if __name__ == "__main__":
    initialize_and_run_workflow()
