function rate(stars) {
    const spans = document.querySelectorAll('.rating span');
    document.getElementById('rating_value').value = stars;
    spans.forEach((span, index) => {
        if (index < stars) {
            span.innerHTML = '&#9733;'; // filled star
            span.classList.add('active');

        } else {
            span.innerHTML = '&#9734;'; // empty star
            span.classList.remove('active');
        }
    });
}

function scrollToBottom() {
    const scrollArrow = document.getElementById('scroll-arrow');
    scrollArrow.classList.add('hidden');
    window.scrollTo({
        top: document.body.scrollHeight,
        behavior: 'smooth'
    });
}

function checkScroll() {
    const scrollArrow = document.getElementById('scroll-arrow');
    const targetElement = document.getElementById('challenge-submit'); // Replace with your element's ID
    const relative_bottom = targetElement.getBoundingClientRect().bottom;

    if ((relative_bottom - 60) < window.innerHeight) {
        scrollArrow.classList.add('hidden');
    } else {
        scrollArrow.classList.remove('hidden')
    }
}

function validateForm() {
    const ratingValue = document.getElementById('rating_value').value;
    if (!ratingValue) {
        alert("Please select a rating before submitting.");
        return false; // Prevent form submission
    }
    return true; // Allow form submission
}

window.addEventListener('resize', checkScroll);
window.addEventListener('scroll', checkScroll); 
document.addEventListener('DOMContentLoaded', checkScroll);

document.addEventListener('DOMContentLoaded', () => {
    // Wait for 2 seconds
    setTimeout(() => {
        const progressContainer = document.getElementById('progress-container');
        progressContainer.style.display = 'none';

        const response = document.getElementsByClassName("response");
        response[0].style.display = "inline";
        checkScroll(); //if the page size increases because we are displaying the prompt, we need to show the scroll-arrow again. 
    }, 1500); // 1500 milliseconds = 1.5 seconds
});

$(document).ready(function() {
    $('#subscribe-btn').on('click', function() {
        var email = $('#email-input').val();
        
        if (email === "") {
            alert("Please enter an email address.");
            return;
        }
        
        $.ajax({
            url: '/subscribe',  // Replace with your server endpoint
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ email: email }),
            success: function(response) {
                $('#subscribe-btn').text('Subscribed').addClass('subscribed').prop('disabled', true);
                $('#email-input').val('');  // Clear the input field
                $('#email-input').attr('placeholder', 'You have successfully subscribed!');
            },
            error: function(xhr, status, error) {
                alert('Subscription failed: ' + error);
            }
        });
    });
});
