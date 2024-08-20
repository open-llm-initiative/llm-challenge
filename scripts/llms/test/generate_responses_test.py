import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from generate_responses import sanitize_response_to_html_and_trim

response = r"""In the following sentences, underline the verb in parentheses that agrees with the collective noun. 

                <strong>Example 1</strong>. The audience <em>(is, $\underline{\text{are}}$)</em> slowly finding their seats in the theater.

                The jury <em>(is, are)</em> deliberating the case.

                In the following sentence, underline the correct modifier from the pair given in parentheses. Example 1. Last weekend we had a (real, $\underline{\text{really}}$) good time.

                The new movie <em>The Matrix</em> is <em>(real, really)</em> exciting.
                                            
                In the following sentence, underline the correct modifier from the pair given in parentheses. Example 1. Last weekend we had a (real, $\underline{\text{really}}$) busy weekend.

                The new movie <em>The Matrix</em> is <em>(real, really)</em> exciting."""

def test_sanitize_response_to_html_and_trim(response):
    response_in_html = None
    try:
        response_in_html = sanitize_response_to_html_and_trim(response=response) 
    except Exception as e:
        print(e)
        raise e

    return True if response_in_html is not None else False

if __name__ == "__main__":
    print ("Running HTML sanitization tests")
    print(f"Sanitize basic text: {test_sanitize_response_to_html_and_trim(response)}")