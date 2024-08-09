function rate(stars) {
    const spans = document.querySelectorAll('.rating span');
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
    if ((window.innerHeight + window.scrollY) +1 < document.body.scrollHeight) {
        scrollArrow.classList.remove('hidden');
    } else {
        scrollArrow.classList.add('hidden');
    }
}

function validateForm() {
    const ratingValue = document.getElementById('rating-value').value;
    if (!ratingValue) {
        alert("Please select a rating before submitting.");
        return false; // Prevent form submission
    }
    return true; // Allow form submission
}

window.addEventListener('resize', checkScroll);
document.addEventListener('DOMContentLoaded', checkScroll);