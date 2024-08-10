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
    if ((window.innerHeight + window.scrollY) + 50 < document.body.scrollHeight) {
        scrollArrow.classList.remove('hidden');
    } else {
        scrollArrow.classList.add('hidden');
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
