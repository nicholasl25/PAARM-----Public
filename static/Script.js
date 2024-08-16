document.addEventListener('DOMContentLoaded', function() {
    setShortTermMode();  // Set default mode to Short Term (Month)

    // Event listeners for buttons
    document.getElementById('shortTermButton').addEventListener('click', setShortTermMode);
    document.getElementById('longTermButton').addEventListener('click', setLongTermMode);
});

function validateForm() {
    var formElement = document.getElementById('settingsForm');
    var statusElement = document.getElementById('status');
    
    var isFutureSelected = formElement.querySelector('input[name="isFuture"]:checked');
    var reportName = formElement.querySelector('input[name="report_name"]').value.trim();
    var months = formElement.querySelectorAll('input[name="month[]"]');
    var years = formElement.querySelectorAll('select[name="year[]"]');
    var selectedModels = formElement.querySelectorAll('input[name="models"]:checked');

    if (!isFutureSelected) {
        statusElement.innerText = 'Please select either Extrapolate or Backtest.';
        statusElement.style.color = 'red';
        return false;
    }
    if (reportName === '') {
        statusElement.innerText = 'Please enter a report name.';
        statusElement.style.color = 'red';
        return false;
    }

    // Check if at least one month or year is selected
    const hasSelectedMonth = Array.from(months).some(month => month.value !== '');
    const hasSelectedYear = Array.from(years).some(year => year.value !== '');

    if (!hasSelectedMonth && !hasSelectedYear) {
        statusElement.innerText = 'Please select at least one month or year.';
        statusElement.style.color = 'red';
        return false;
    }

    // Skip model check if Deep Learning is selected
    if (deepLearner.value === 'False' && selectedModels.length === 0) {
        statusElement.innerText = 'Please select at least one model.';
        statusElement.style.color = 'red';
        return false;
    }

    const includeCovid = document.getElementById('includeCovid').checked;
    const covidStart = document.getElementById('covidStart').value;
    const covidEnd = document.getElementById('covidEnd').value;

    if (includeCovid) {
        if (covidStart === '' || covidEnd === '') {
            statusElement.innerText = 'Please select both COVID start and end dates.';
            statusElement.style.color = 'red';
            return false;
        }
        if (covidEnd < covidStart) {
            statusElement.innerText = 'COVID end date must be after start date.';
            statusElement.style.color = 'red';
            return false;
        }
        if (new Date(covidStart).getFullYear() < 2019) {
            statusElement.innerText = 'COVID start date must be after 2019.';
            statusElement.style.color = 'red';
            return false;
        }
        if (new Date(covidEnd).getFullYear() >= 2023) {
            statusElement.innerText = 'COVID end date must be before 2024.';
            statusElement.style.color = 'red';
            return false;
        }

    }

    return true;
}

function toggleAdvancedSettings() {
var content = document.getElementById("advanced-settings-content");
content.style.display = (content.style.display === "none") ? "block" : "none";
}

function toggleLongTermMode(isLongTerm) {
    var startInput = document.getElementById("covidStart");
    var endInput = document.getElementById("covidEnd");

    if (isLongTerm) {
        startInput.type = "number";
        startInput.placeholder = "YYYY";
        startInput.min = "1900";
        startInput.max = "2100";

        endInput.type = "number";
        endInput.placeholder = "YYYY";
        endInput.min = "1900";
        endInput.max = "2100";
    } else {
        startInput.type = "month";
        endInput.type = "month";
    }
}

function submitForm(event) {
    event.preventDefault();
    if (!validateForm()) {
        return;
    }

    var formData = new FormData(document.getElementById('settingsForm'));
    var formElement = document.getElementById('settingsForm');
    var statusElement = document.getElementById('status');
    
    // Hide the form
    formElement.style.display = 'none';
    
    // Show the running status
    statusElement.innerText = 'Running Report...';
    statusElement.style.fontSize = '24px';  // Make text bigger
    statusElement.style.color = '#333';
    
    fetch('/process', {
        method: 'POST',
        body: formData
    })
    .then(response => response.text())
    .then(data => {
        if (data.includes("Complete")) {
            statusElement.innerText = 'Report Successfully Ran!';
            statusElement.style.color = 'green';
        } else {
            statusElement.innerText = 'Error: ' + data;
            statusElement.style.color = 'red';
        }
        statusElement.style.fontSize = '24px';
        setTimeout(function() { window.close(); }, 5000);
    })
    .catch(error => {
        statusElement.innerText = 'Error: ' + error;
        statusElement.style.color = 'red';
    });
}

function setTerm(term) {
    // Clear all existing inputs
    if (term === 'short') {
        // Show month inputs and button, hide year inputs and button
        document.querySelectorAll('.yearInput').forEach(input => input.parentNode.remove());
        document.getElementById('monthInput').style.display = 'block';
        document.getElementById('addMonthButton').style.display = 'inline-block';
        document.getElementById('yearInput').style.display = 'none';
        document.getElementById('addYearButton').style.display = 'none';
    } else if (term === 'long') {
        // Show year inputs and button, hide month inputs and button
        document.querySelectorAll('.monthInput').forEach(input => input.parentNode.remove());
        document.getElementById('monthInput').style.display = 'none';
        document.getElementById('addMonthButton').style.display = 'none';
        document.getElementById('yearInput').style.display = 'block';
        document.getElementById('addYearButton').style.display = 'inline-block';
    }
}

function addMonthInput() {
  const container = document.getElementById('timeInputsContainer');
  const wrapper = document.createElement('div');
  wrapper.classList.add('timeInputWrapper');
  const isFirstMonth = container.children.length === 0; // Check if it's the first input
  wrapper.innerHTML = `
      <div class="monthInput">
          <label for="month">Month:</label>
          <input type="month" name="month[]">
          ${isFirstMonth ? '' : '<button type="button" onclick="removeTimeInput(this)">Remove</button>'}
      </div>
  `;
  container.appendChild(wrapper);
}

function addYearInput() {
  const container = document.getElementById('timeInputsContainer');
  const wrapper = document.createElement('div');
  wrapper.classList.add('timeInputWrapper');
  const isFirstYear = container.children.length === 0; // Check if it's the first input
  wrapper.innerHTML = `
      <div class="yearInput">
          <label for="year">Year:</label>
          <select name="year[]">
              <!-- Options will be populated by JavaScript -->
          </select>
          ${isFirstYear ? '' : '<button type="button" onclick="removeTimeInput(this)">Remove</button>'}
      </div>
  `;
  container.appendChild(wrapper);
  populateYearOptions(wrapper.querySelector('select'));
}
  
function removeTimeInput(button) {
    button.closest('.timeInputWrapper').remove();
}

function toggleAirports() {
    const allAirportsCheckbox = document.getElementById('allAirports');
    const airportCheckboxes = document.querySelectorAll('.airport-checkbox');
    const airportSubtotal = document.getElementById('airportSubtotal');

    if (allAirportsCheckbox.checked) {
        airportCheckboxes.forEach(checkbox => {
            checkbox.disabled = true;
            checkbox.checked = true;
        });
        airportSubtotal.disabled = true;
        airportSubtotal.checked = true;
    } else {
        airportCheckboxes.forEach(checkbox => {
            checkbox.disabled = false;
            checkbox.checked = false;
        });
        airportSubtotal.disabled = false;
        airportSubtotal.checked = false;
    }
}

function toggleDomInt() {
    const bothCheckbox = document.getElementById('both');
    const domIntCheckboxes = document.querySelectorAll('.domint-checkbox');
    const domIntSubtotal = document.getElementById('domIntSubtotal');

    if (bothCheckbox.checked) {
        domIntCheckboxes.forEach(checkbox => {
            checkbox.disabled = true;
            checkbox.checked = true;
        });
        domIntSubtotal.disabled = true;
        domIntSubtotal.checked = true;
    } else {
        domIntCheckboxes.forEach(checkbox => {
            checkbox.disabled = false;
            checkbox.checked = false;
        });
        domIntSubtotal.disabled = false;
        domIntSubtotal.checked = false;
    }
}

let isLongTerm = false; // Variable to track the state

function setShortTermMode() {
    isLongTerm = false; // Set state to short term
    document.getElementById('shortTermButton').classList.add('active');
    document.getElementById('shortTermButton').classList.remove('inactive');
    document.getElementById('longTermButton').classList.add('inactive');
    document.getElementById('longTermButton').classList.remove('active');
    
    document.getElementById('monthInput').style.display = 'block';
    document.getElementById('yearInput').style.display = 'none';

    document.getElementById('long_term').value = 'False';
}

function setLongTermMode() {
    isLongTerm = true; // Set state to long term
    document.getElementById('longTermButton').classList.add('active');
    document.getElementById('longTermButton').classList.remove('inactive');
    document.getElementById('shortTermButton').classList.add('inactive');
    document.getElementById('shortTermButton').classList.remove('active');
    
    document.getElementById('monthInput').style.display = 'none';
    document.getElementById('yearInput').style.display = 'block';

    document.getElementById('long_term').value = 'True';
}

// Example function to populate year options
function populateYearOptions(selectElement) {
    const currentYear = new Date().getFullYear();
    const startYear = 2000;
    const endYear = currentYear + 10;

    for (let year = startYear; year <= endYear; year++) {
        const option = document.createElement('option');
        option.value = year;
        option.textContent = year;
        selectElement.appendChild(option);
    }
}

function toggleCovidFields() {
    var checkbox = document.getElementById('includeCovid');
    var startField = document.getElementById('covidStart');
    var endField = document.getElementById('covidEnd');
    
    if (checkbox.checked) {
        startField.disabled = false;
        endField.disabled = false;
        startField.classList.remove('disabled-input');
        endField.classList.remove('disabled-input');
    } else {
        startField.disabled = true;
        endField.disabled = true;
        startField.classList.add('disabled-input');
        endField.classList.add('disabled-input');
    }
}

function toggleTermMode(isLongTerm) {
    const monthButton = document.getElementById('addMonthButton');
    const yearButton = document.getElementById('addYearButton');
    const monthInput = document.getElementById('monthInput');
    const yearInput = document.getElementById('yearInput');

    if (isLongTerm) {
        monthButton.style.display = 'none';
        yearButton.style.display = 'block';
        monthInput.style.display = 'none';
        yearInput.style.display = 'block';
    } else {
        monthButton.style.display = 'block';
        yearButton.style.display = 'none';
        monthInput.style.display = 'block';
        yearInput.style.display = 'none';
    }
}

// Example usage
document.getElementById('addMonthButton').onclick = function() {
    toggleTermMode(false);
};
document.getElementById('addYearButton').onclick = function() {
    toggleTermMode(true);
};
function toggleView(view) {
    var sarimaxSection = document.getElementById('sarimaxSection');
    var deepLearningSection = document.getElementById('deepLearningSection');
    var sarimaxButton = document.getElementById('sarimaxButton');
    var deepLearningButton = document.getElementById('deepLearningButton');
    var deepLearner = document.getElementById('deepLearner');

    if (view === 'sarimax') {
        sarimaxSection.classList.remove('hidden');
        deepLearningSection.classList.add('hidden');
        sarimaxButton.classList.add('active');
        sarimaxButton.classList.remove('inactive');
        deepLearningButton.classList.add('inactive');
        deepLearningButton.classList.remove('active');
        deepLearner.value = 'False';
    } else if (view === 'deepLearning') {
        sarimaxSection.classList.add('hidden');
        deepLearningSection.classList.remove('hidden');
        sarimaxButton.classList.add('inactive');
        sarimaxButton.classList.remove('active');
        deepLearningButton.classList.add('active');
        deepLearningButton.classList.remove('inactive');
        deepLearner.value = 'True';
    }
}

// Initialize with SARIMAX view
toggleView('sarimax');
// Initialize the buttons based on default term
setTerm('short'); // or 'long', based on your default choice