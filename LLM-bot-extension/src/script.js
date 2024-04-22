let observer = null;
var llmResponse = "";
var currentComment;
var url = 'http://127.0.0.1:80/';
var promptID = null;
var token = '';
var googleGemma2b = 'google/gemma-2b-it';
var stabilityAi2b = 'stabilityai/stable-code-instruct-3b';
var tinyLlama = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0';
const defaultLLMs = ['google/gemma-2b-it', 'stabilityai/stable-code-instruct-3b', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'];


// ---------- LISTENERS ----------

window.addEventListener('popstate', setupNav);
window.addEventListener('hashchange', setupNav);
document.addEventListener('pjax:end', setupNav);

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupNav);
} else {
    setupNav();
}

// Extension listener
document.addEventListener('DOMContentLoaded', async function () {
    //Check if on github
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
        if (tabs && tabs[0] && tabs[0].url) {
            const currentURL = tabs[0].url;
            if (currentURL.includes("github.com")) {
                document.getElementById('content').style.display = 'block';
                updateHuggingFaceTokenIcon();
                updateGitHubTokenIcon();
                updateLoadLLMIcon();
                updateSelectedLLMInDropdown();
            } else {
                document.getElementById('message').style.display = 'block';
            }
        } else {
            console.error("No tabs were found.");
        }
    });

    //Add settings
    addReformToggle();
    addStatusToggle();
    addReviewsToggle()
    addCodeToggle()
    addRelevanceToggle()
    addToxicToggle()
    initializeLlmDropdown()
    addEventLoadLLM()
    addEventDeleteLLM()
    addEventGithubSaveToken()
    addEventHuggingFaceSaveToken()
    addEventSaveCustomLLM()
});

// Toggle state listener
chrome.runtime.onMessage.addListener(async function (request, sender, sendResponse) {
    let icon = document.getElementById('LLM_Icon');
    //Update toggle state
    if (icon && (request.toggleState == true || request.toggleState == false)) {
        icon.style.display = request.toggleState ? 'block' : 'none';
        let state = request.toggleState ? "checked" : "not_checked";
        try {
            await chrome.storage.sync.set({ 'toggleState': state });
            updateIconVisibility()

        } catch (error) {
            console.log("Error:", error);
        }
    }

    //Handle toggle reform comment state
    let copyButton = document.getElementById("copySuggestion");
    if (copyButton && request.toggleReform != null) {
        if (request.toggleReform) {
            copyButton.style.display = "flex";
        } else if (!request.toggleReform) {
            copyButton.style.display = "none";
        }
    }

    //Update Icon when changing tabs 
    if (request.action === "updateIconOnTabChange") {
        updateIconVisibility();
        sendResponse({ result: "UpdatedIcon" });

    }

    return true;
});




// ---------- GENERAL FUNCTIONS ----------

// Watch changes in the DOM
function observeDOM() {
    if (observer) {
        observer.disconnect();
    }
    observer = new MutationObserver(mutations => {
        mutations.forEach(mutation => {
            if (mutation.type === "childList" && mutation.addedNodes.length) {
                checkIcon();
                attachCopyEvent();
            }
        });
    });
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
}

// Watch changed when selecting differents tabs in github Single Application Page (SAP)
async function setupNav() {
    observeDOM();
    checkIcon();
    attachTextAreaEvent();
    attachCopyEvent();
    updateIconVisibility();
}




// ---------- ICONS FUNCTIONS ----------

// Check if icon exists
function checkIcon() {
    const newCommentField = document.getElementById("new_comment_field");
    let icon = document.getElementById("LLM_Icon");
    if (newCommentField && !icon) {
        addIconOverCommentBox();
        createRequestSectionStructure();
        createResponseSectionStructure();
    }
    if (icon) {
        attachIconEvent(icon);
    }
}

// Add Icon over github comment box
async function addIconOverCommentBox() {
    let textarea = document.getElementById("new_comment_field");
    if (textarea) {
        try {
            //add a function to add the text area
            //add_LLM_Reply_Area();
            //Creating the LLM icon
            let icon = document.createElement('img');
            icon.id = "LLM_Icon";
            icon.alt = "Icon";
            icon.src = chrome.runtime.getURL("icon/logo-nath-48.png");



            icon.onmouseover = function () {
                this.style.transform = "translateY(-10%) scale(1.1)";
            };
            icon.onmouseout = function () {
                this.style.transform = "translateY(-10%) scale(1)";
            };

            icon.style.cssText = `
                pointer-events: auto;
                cursor: pointer;
                position: absolute;
                right: 10px;
                top: 50%;
                transform: translateY(-10%);
                z-index: 1000;`
                ;

            attachIconEvent(icon);

            textarea.parentElement.style.position = "relative";
            textarea.parentElement.appendChild(icon);
            updateIconVisibility();
        } catch (error) {
        }
    }
}

// LLMResponse setter
function saveLLMResponse(input) {
    llmResponse = input;
}

// LLMResponse getter
function getLLMResponse() {
    return llmResponse;
}

// Get the response from local storage
async function getResponse() {
    return new Promise((resolve, reject) => {
        try {
            chrome.storage.local.get('llmResponse', function (result) {
                if (chrome.runtime.lastError) {
                    reject(chrome.runtime.lastError);
                } else {
                    resolve(result['llmResponse']);
                }
            });
        } catch (error) { }
    });
}

function createResponseSectionStructure() {
    let commentForm = document.getElementById("new_comment_form");
    if (commentForm) {
        const modal = document.createElement('section');
        modal.classList.add('modal');
        modal.classList.add('hidden');
        modal.setAttribute("id", "response-modal");

        const modalContent = document.createElement('div');
        modalContent.classList.add('flex');
        modalContent.classList.add('response-content');

        // Adding title and textarea for each element
        const elements = ['Toxicity', 'Relevance', 'Reformulation'];
        elements.forEach(element => {
            const title = document.createElement('h3');
            title.textContent = element;
            modalContent.appendChild(title);

            const textarea = document.createElement('textarea');
            const textareaId = element.toLowerCase().replace(/\s+/g, '-') + '_response_area'; // Convert element name to lowercase and replace spaces with hyphens
            textarea.id = textareaId; // Set textarea ID
            textarea.setAttribute("readonly", true); // Make textarea readonly
            textarea.style.width = '100%';
            textarea.style.resize = 'none'; // Disable resizing
            textarea.style.marginBottom = '10px'; // Add margin between each textarea
            textarea.style.padding = '8px'; // Add padding to textarea
            textarea.style.borderRadius = '4px'; // Add border radius
            textarea.style.border = '1px solid #ccc'; // Add border
            modalContent.appendChild(textarea);
            if (element === 'Reformulation') {
                const copyButton = document.createElement('button');
                copyButton.id = 'response_copy_button';
                copyButton.textContent = 'Copy';
                copyButton.classList.add('copy-button');
                copyButton.style.marginBottom = '10px'; // Add margin below the button
                copyButton.style.backgroundColor = 'blue'; // Set button background color to blue
                copyButton.style.color = '#fff'; // Set button text color to white
                copyButton.style.border = 'none'; // Remove button border
                copyButton.style.borderRadius = '4px'; // Add border radius
                copyButton.style.padding = '8px 16px'; // Set button padding
                copyButton.style.width = 'auto'; // Set button width to auto
                copyButton.style.display = 'inline-block'; // Make the button inline-block
                copyButton.onclick = async function () {
                    const textarea = document.getElementById(textareaId);
                    try {
                        await navigator.clipboard.writeText(textarea.value);
                        console.log('Text copied to clipboard');
                    } catch (error) {
                        console.error('Failed to copy text: ', error);
                    }
                };
                modalContent.appendChild(copyButton);
            }

        });

        modal.appendChild(modalContent);

        const overlay = document.createElement('div');
        overlay.classList.add('overlay');
        overlay.classList.add('hidden');
        overlay.setAttribute("id", "response-overlay");
        overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.4)'; // Add background color

        commentForm.parentElement.appendChild(modal);
        commentForm.parentElement.appendChild(overlay);
        modal.setAttribute("style", "display: none;");
    }
}

async function fillResponseSection(promptsResponses) {


    // Toxicity
    const toxicityTextarea = document.getElementById('toxicity_response_area');
    if (toxicityTextarea) {
        const toxicState = await getToggleState('toggleToxicity');
        if (toxicState === 'checked') {
            let toxicityResponse = promptsResponses.get("Toxicity");
            console.log("toxictyResponse: " + toxicityResponse);
            toxicityTextarea.value = toxicityResponse;
            toxicityTextarea.setAttribute("style", "display: block; width: 100%;"); // Show the textarea
            const toxicityTitle = toxicityTextarea.previousElementSibling;
            if (toxicityTitle) {
                toxicityTitle.setAttribute("style", "display: block; width: 100%;"); // Show the title
            }
        } else {
            toxicityTextarea.setAttribute("style", "display: none; width: 100%;"); // Hide the textarea
            const toxicityTitle = toxicityTextarea.previousElementSibling;
            if (toxicityTitle) {
                toxicityTitle.setAttribute("style", "display: none; width: 100%;"); // Hide the title
            }
        }
    }


    // relevance
    const relevanceTextarea = document.getElementById('relevance_response_area');
    if (relevanceTextarea) {
        const relevanceState = await getToggleState('toggleRelevance');
        if (relevanceState === 'checked') {
            relevanceTextarea.value = promptsResponses.get("Relevance");
            relevanceTextarea.setAttribute("style", "display: block; width: 100%;"); // Show the textarea
            const relevanceTitle = relevanceTextarea.previousElementSibling;
            if (relevanceTitle) {
                relevanceTitle.setAttribute("style", "display: block; width: 100%;"); // Show the title
            }
        } else {
            relevanceTextarea.setAttribute("style", "display: none; width: 100%;"); // Hide the textarea
            const relevanceTitle = relevanceTextarea.previousElementSibling;
            if (relevanceTitle) {
                relevanceTitle.setAttribute("style", "display: none; width: 100%;"); // Hide the title
            }
        }
    }

    // Reformulation
    const reformTextarea = document.getElementById('reformulation_response_area');

    if (reformTextarea) {
        const reformState = await getToggleState('toggleReform');
        if (reformState === 'checked') {
            reformTextarea.value = promptsResponses.get("Reformulation");
            reformTextarea.setAttribute("style", "display: block; width: 100%;"); // Show the textarea
            const reformTitle = reformTextarea.previousElementSibling;
            if (reformTitle) {
                reformTitle.setAttribute("style", "display: block; width: 100%;"); // Show the title
            }
            const copyButton = document.getElementById('response_copy_button');
            if (copyButton) {
                copyButton.setAttribute("style", "display: block; width: 100%; margin-bottom: 10px; background-color: blue; color: #fff; border: none; border-radius: 4px; padding: 8px 16px; width: auto; display: inline-block;"); // Hide the title
            }
        } else {
            reformTextarea.setAttribute("style", "display: none;"); // Hide the textarea
            const reformTitle = reformTextarea.previousElementSibling;
            if (reformTitle) {
                reformTitle.setAttribute("style", "display: none; width: 100%;"); // Hide the title
            }
            const copyButton = document.getElementById('response_copy_button');
            if (copyButton) {
                copyButton.setAttribute("style", "display: none;"); // Hide the title
            }
        }
    }
}



function createRequestSectionStructure() {
    let commentForm = document.getElementById("new_comment_form");
    if (commentForm) {
        const modal = document.createElement('section');
        modal.classList.add('modal');
        modal.classList.add('hidden');
        modal.setAttribute("id", "request-modal");

        const modalContent = document.createElement('div');
        modalContent.classList.add('flex');
        modalContent.classList.add('request-content');

        // Adding title and textarea for each element
        const elements = ['Files', 'Reviews', 'General Prompt', 'Toxicity', 'Relevance', 'Reformulation'];
        elements.forEach(element => {
            const title = document.createElement('h3');
            title.textContent = element;
            modalContent.appendChild(title);

            const textarea = document.createElement('textarea');
            const textareaId = element.toLowerCase().replace(/\s+/g, '-') + '_prompt_area'; // Convert element name to lowercase and replace spaces with hyphens
            textarea.id = textareaId; // Set textarea ID
            textarea.style.width = '100%';
            textarea.style.resize = 'both'; // Allow both horizontal and vertical resizing
            textarea.style.marginBottom = '10px'; // Add margin between each textarea
            textarea.style.padding = '8px'; // Add padding to textarea
            textarea.style.borderRadius = '4px'; // Add border radius
            textarea.style.border = '1px solid #ccc'; // Add border
            modalContent.appendChild(textarea);
        });

        const sendButton = document.createElement('button');
        sendButton.textContent = 'Send to LLM';
        sendButton.id = 'send_to_LLM_button'; // Set button ID
        sendButton.classList.add('send-button');
        sendButton.style.backgroundColor = 'green'; // Set button background color
        sendButton.style.color = '#fff'; // Set button text color
        sendButton.style.padding = '10px 20px'; // Set button padding
        sendButton.style.border = 'none'; // Remove button border
        sendButton.style.borderRadius = '4px'; // Add border radius
        sendButton.style.marginTop = '20px'; // Add margin top
        sendButton.style.marginLeft = 'auto'; // Align button to the center
        sendButton.style.marginRight = 'auto'; // Align button to the center
        sendButton.style.display = 'inline-block'; // Make the button an inline-block element
        sendButton.onclick = async function () {
            const request_modal = document.getElementById('request-modal');
            request_modal.setAttribute("style", "display: none;");
            commentForm.innerHTML += '<div id="loading-response-icon">Sending request to LLM... ⏳</div>';
            let promptsResponses = await createPrompts();
            const loadingIcon = document.getElementById('loading-response-icon')
            if (loadingIcon) {
                loadingIcon.remove();
            }
            const response_modal = document.getElementById('response-modal');
            fillResponseSection(promptsResponses);
            response_modal.setAttribute("style", "display: block;");
        };
        modalContent.appendChild(sendButton);

        modal.appendChild(modalContent);

        const overlay = document.createElement('div');
        overlay.classList.add('overlay');
        overlay.classList.add('hidden');
        overlay.setAttribute("id", "request-overlay");
        overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.4)'; // Add background color

        commentForm.parentElement.appendChild(modal);
        commentForm.parentElement.appendChild(overlay);
        modal.setAttribute("style", "display: none;");
    }
}


async function fillRequestSection() {
    let modal = document.getElementById("request-modal");
    let overlay = document.getElementById("request-overlay");
    let modalContent = document.getElementById("request-content");

    const generalPromptTextarea = document.getElementById('general-prompt_prompt_area');
    if (generalPromptTextarea) {
        let basePrompt = await getBasePrompt();
        // Set the textarea content
        generalPromptTextarea.value = basePrompt;
        generalPromptTextarea.setAttribute("style", "display: block; width: 100%;");
        const generalPromptTitle = generalPromptTextarea.previousElementSibling;
        if (generalPromptTitle) {
            generalPromptTitle.setAttribute("style", "display: block; width: 100%;"); // Show the title
        }
    }
    // Load the prompts into each textarea to allow the user to modify them before sending

    // Files
    const filesTextarea = document.getElementById('files_prompt_area');
    if (filesTextarea) {
        const codeState = await getToggleState('toggleCode');
        if (codeState === 'checked') {
            //todo: set text
            let filePrompt = await getFilePrompt(); //files_prompt_area
            filesTextarea.setAttribute("style", "display: block; width: 100%;");
            filesTextarea.value = filePrompt;
            const filesTitle = filesTextarea.previousElementSibling;
            if (filesTitle) {
                filesTitle.setAttribute("style", "display: block; width: 100%;");; // Show the title
            }
        } else {
            filesTextarea.setAttribute("style", "display: none; width: 100%;"); // Hide the textarea
            const filesTitle = filesTextarea.previousElementSibling;
            if (filesTitle) {
                filesTitle.setAttribute("style", "display: none; width: 100%;"); // Hide the title
            }
        }
    }

    // Reviews
    const reviewsTextarea = document.getElementById('reviews_prompt_area');
    if (reviewsTextarea) {
        const reviewState = await getToggleState('toggleReviews');
        if (reviewState === 'checked') {
            let reviewPrompt = await getReviewPrompt();
            reviewsTextarea.setAttribute("style", "display: block; width: 100%;"); // Show the textarea
            reviewsTextarea.value = reviewPrompt;
            const reviewsTitle = reviewsTextarea.previousElementSibling;
            if (reviewsTitle) {
                reviewsTitle.setAttribute("style", "display: block; width: 100%;"); // Show the title
            }
        } else {
            reviewsTextarea.setAttribute("style", "display: none; width: 100%;"); // Hide the textarea
            const reviewsTitle = reviewsTextarea.previousElementSibling;
            if (reviewsTitle) {
                reviewsTitle.setAttribute("style", "display: none; width: 100%;"); // Hide the title
            }
        }
    }

    // Toxicity
    const toxicityTextarea = document.getElementById('toxicity_prompt_area');
    if (toxicityTextarea) {
        const toxicState = await getToggleState('toggleToxicity');
        if (toxicState === 'checked') {
            let toxicityPrompt = getToxicityPrompt();
            toxicityTextarea.setAttribute("style", "display: block; width: 100%;"); // Show the textarea
            toxicityTextarea.value = toxicityPrompt;
            const toxicityTitle = toxicityTextarea.previousElementSibling;
            if (toxicityTitle) {
                toxicityTitle.setAttribute("style", "display: block; width: 100%;"); // Show the title
            }
        } else {
            toxicityTextarea.setAttribute("style", "display: none; width: 100%;"); // Hide the textarea
            const toxicityTitle = toxicityTextarea.previousElementSibling;
            if (toxicityTitle) {
                toxicityTitle.setAttribute("style", "display: none; width: 100%;"); // Hide the title
            }
        }
    }


    // relevance
    const relevanceTextarea = document.getElementById('relevance_prompt_area');
    if (relevanceTextarea) {
        const toxicState = await getToggleState('toggleRelevance');
        if (toxicState === 'checked') {
            let relevancePrompt = getRelevancePrompt();
            relevanceTextarea.setAttribute("style", "display: block; width: 100%;"); // Show the textarea
            relevanceTextarea.value = relevancePrompt;
            const relevanceTitle = relevanceTextarea.previousElementSibling;
            if (relevanceTitle) {
                relevanceTitle.setAttribute("style", "display: block; width: 100%;"); // Show the title
            }
        } else {
            relevanceTextarea.setAttribute("style", "display: none; width: 100%;"); // Hide the textarea
            const relevanceTitle = relevanceTextarea.previousElementSibling;
            if (relevanceTitle) {
                relevanceTitle.setAttribute("style", "display: none; width: 100%;"); // Hide the title
            }
        }
    }

    // Reformulation
    const reformTextarea = document.getElementById('reformulation_prompt_area');
    if (reformTextarea) {
        const reformState = await getToggleState('toggleReform');
        if (reformState === 'checked') {
            let reformPrompt = getReformPrompt();
            reformTextarea.setAttribute("style", "display: block; width: 100%;"); // Show the textarea
            reformTextarea.value = reformPrompt;
            const reformTitle = reformTextarea.previousElementSibling;
            if (reformTitle) {
                reformTitle.setAttribute("style", "display: block; width: 100%;"); // Show the title
            }
        } else {
            reformTextarea.setAttribute("style", "display: none; width: 100%;"); // Hide the textarea
            const reformTitle = reformTextarea.previousElementSibling;
            if (reformTitle) {
                reformTitle.setAttribute("style", "display: none; width: 100%;"); // Hide the title
            }
        }
    }
}


// Attach onclick event on LLM Icon
function attachIconEvent(icon) {
    icon.onmouseover = function () {
        this.style.transform = "translateY(-10%) scale(1.1)";
    };
    icon.onmouseout = function () {
        this.style.transform = "translateY(-10%) scale(1)";
    };
    icon.onclick = async function (event) {

        const request_modal = document.getElementById('request-modal');
        request_modal.setAttribute("style", "display: block;");

        const response_modal = document.getElementById('response-modal');
        response_modal.setAttribute("style", "display: none;");
        fillRequestSection();
        event.stopPropagation();
        updateIconVisibility();
    };
}

// Update the visibility of the LLM Icon
async function updateIconVisibility() {
    var icon = document.getElementById("LLM_Icon");
    var textarea = document.getElementById("popup-llm")
    let currrentState = "";
    if (icon) {
        try {
            currrentState = await getToggleState('toggleState');
            if (currrentState === "checked") {
                icon.style.display = 'block';
            } else {
                icon.style.display = 'none';
                textarea.style.display = 'none';
            }
        } catch (error) {
            console.log("Error:", error);
        }
    }
}



// ---------- TOGGLES FUNCTIONS ----------

// Get the state enable/disable of the extension
async function getToggleState(key) {
    return new Promise((resolve, reject) => {
        try {
            chrome.storage.sync.get([key], function (result) {
                if (chrome.runtime.lastError) {
                    reject(chrome.runtime.lastError);
                } else {
                    resolve(result[key]);
                }
            });
        } catch (error) {
        }
    });
}

// Add reformulate toggle
async function addReformToggle() {
    let lswitch = document.getElementById('reformulationSwitch');
    let currrentState = "";
    let toggle = document.createElement("input");

    try {
        currrentState = await getToggleState('toggleReform');
        if (!currrentState) {
            chrome.runtime.sendMessage({
                from: 'popup',
                subject: 'toggleReform',
                toggleReform: toggle.checked
            });
            currrentState = await getToggleState('toggleReform');
        }
    } catch (error) {
        console.log("Error:", message);
    }

    
    toggle.type = "checkbox";
    toggle.id = "toggleReform";

    let slider = document.createElement("span");
    slider.classList = "slider round";

    if (lswitch) {
        if (currrentState === "checked") {
            toggle.checked = true;
        } else {
            toggle.checked = false;
        }
        lswitch.appendChild(toggle);
        lswitch.appendChild(slider);
        toggle.addEventListener('change', function () {
            chrome.runtime.sendMessage({
                from: 'popup',
                subject: 'toggleReform',
                toggleReform: toggle.checked
            });
        });
    }
}

// asdjsnajncvjnakcasnjcnjasnc



// dksancnsanjcnsajncjasnjcjsancjsa






// nsjdnjasndjsandjandjansjdnasj

// Add enable/disable toggle
async function addStatusToggle() {
    let lswitch = document.getElementById('enableSwitch');
    let currrentState = "";
    let toggle = document.createElement("input");

    try {
        currrentState = await getToggleState('toggleState');
        if (!currrentState) {
            chrome.runtime.sendMessage({
                from: 'popup',
                subject: 'toggleState',
                toggleState: toggle.checked
            });
            currrentState = await getToggleState('toggleState');
        }
    } catch (error) {
        console.log("Error:", message);
    }

    toggle.type = "checkbox";
    toggle.id = "toggleExtension";

    let slider = document.createElement("span");
    slider.classList = "slider round";

    if (lswitch) {
        if (currrentState === "checked") {
            toggle.checked = true;
            document.getElementById('Settings').style.display = "block";
        } else {
            toggle.checked = false;
            document.getElementById('Settings').style.display = "none";
        }
        lswitch.appendChild(toggle);
        lswitch.appendChild(slider);
        toggle.addEventListener('change', function (e) {
            chrome.runtime.sendMessage({
                from: 'popup',
                subject: 'toggleState',
                toggleState: toggle.checked
            });
            if (e.target.checked) {
                document.getElementById('Settings').style.display = "block";
            } else {
                document.getElementById('Settings').style.display = "none";
            }
        });
    }
}

// Add code toggle
async function addCodeToggle() {
    let lswitch = document.getElementById('codeSwitch');
    let currrentState = "";
    let toggle = document.createElement("input");

    try {
        currrentState = await getToggleState('toggleCode');
        if (!currrentState) {
            chrome.runtime.sendMessage({
                from: 'popup',
                subject: 'toggleCode',
                toggleReform: toggle.checked
            });
            currrentState = await getToggleState('toggleCode');
        }
    } catch (error) {
        console.log("Error:", message);
    }

    toggle.type = "checkbox";
    toggle.id = "toggleCode";

    let slider = document.createElement("span");
    slider.classList = "slider round";

    if (lswitch) {
        if (currrentState === "checked") {
            toggle.checked = true;
        } else {
            toggle.checked = false;
        }
        lswitch.appendChild(toggle);
        lswitch.appendChild(slider);
        toggle.addEventListener('change', function (e) {
            chrome.runtime.sendMessage({
                from: 'popup',
                subject: 'toggleCode',
                toggleCode: toggle.checked
            });
            if (e.target.checked) {
                alert("This option can significantly slow down the response generation if there are too many files.");
            }
        });
    }
}

// Add reviews toggle
async function addReviewsToggle() {
    let lswitch = document.getElementById('reviewsSwitch');
    let currrentState = "";
    let toggle = document.createElement("input");

    try {
        currrentState = await getToggleState('toggleReviews');
        if (!currrentState) {
            chrome.runtime.sendMessage({
                from: 'popup',
                subject: 'toggleReviews',
                toggleReform: toggle.checked
            });
            currrentState = await getToggleState('toggleReviews');
        }
    } catch (error) {
        console.log("Error:", message);
    }

    toggle.type = "checkbox";
    toggle.id = "toggleReviews";

    let slider = document.createElement("span");
    slider.classList = "slider round";

    if (lswitch) {
        if (currrentState === "checked") {
            toggle.checked = true;
        } else {
            toggle.checked = false;
        }
        lswitch.appendChild(toggle);
        lswitch.appendChild(slider);
        toggle.addEventListener('change', function () {
            chrome.runtime.sendMessage({
                from: 'popup',
                subject: 'toggleReviews',
                toggleReviews: toggle.checked
            });
        });
    }
}

// Add relevance toggle
async function addRelevanceToggle() {
    let lswitch = document.getElementById('relevanceSwitch');
    let currrentState = "";
    let toggle = document.createElement("input");

    try {
        currrentState = await getToggleState('toggleRelevance');
        if (!currrentState) {
            chrome.runtime.sendMessage({
                from: 'popup',
                subject: 'toggleRelevance',
                toggleReform: toggle.checked
            });
            currrentState = await getToggleState('toggleRelevance');
        }
    } catch (error) {
        console.log("Error:", message);
    }

    toggle.type = "checkbox";
    toggle.id = "toggleRelevance";

    let slider = document.createElement("span");
    slider.classList = "slider round";

    if (lswitch) {
        if (currrentState === "checked") {
            toggle.checked = true;
        } else {
            toggle.checked = false;
        }
        lswitch.appendChild(toggle);
        lswitch.appendChild(slider);
        toggle.addEventListener('change', function () {
            chrome.runtime.sendMessage({
                from: 'popup',
                subject: 'toggleRelevance',
                toggleRelevance: toggle.checked
            });
        });
    }
}

// Add toxicity toggle
async function addToxicToggle() {
    let lswitch = document.getElementById('toxicitySwitch');
    let currrentState = "";
    let toggle = document.createElement("input");

    try {
        currrentState = await getToggleState('toggleToxicity');
        if (!currrentState) {
            chrome.runtime.sendMessage({
                from: 'popup',
                subject: 'toggleToxicity',
                toggleReform: toggle.checked
            });
            currrentState = await getToggleState('toggleToxicity');
        }
    } catch (error) {
        console.log("Error:", message);
    }

    toggle.type = "checkbox";
    toggle.id = "toggleToxicity";

    let slider = document.createElement("span");
    slider.classList = "slider round";

    if (lswitch) {
        if (currrentState === "checked") {
            toggle.checked = true;
        } else {
            toggle.checked = false;
        }
        lswitch.appendChild(toggle);
        lswitch.appendChild(slider);
        toggle.addEventListener('change', function () {
            chrome.runtime.sendMessage({
                from: 'popup',
                subject: 'toggleToxicity',
                toggleToxicity: toggle.checked
            });
        });
    }
}

// ---------- LLM CHANGE FUNCTIONS ----------


function initializeLlmDropdown() {
    chrome.storage.sync.get({ customLlms: [] }, function (result) {
        const llmDropdown = document.getElementById('llm_selected');
        result.customLlms.forEach(llmId => {
            const newOption = new Option(llmId, llmId, false, false);
            llmDropdown.add(newOption, undefined);
        });
    });
}

function updateSelectedLLMInDropdown() {
    const llmDropdown = document.getElementById('llm_selected');
    chrome.storage.sync.get('selectedLlmId', function (result) {
        const selectedLlmId = result.selectedLlmId;
        if (selectedLlmId) {
            llmDropdown.value = selectedLlmId;
        }
    });
}

async function addEventLoadLLM() {
    const button = document.getElementById("llm_change_button");
    button.addEventListener("click", async function () {
        const selectedValue = document.getElementById("llm_selected").value;
        
        chrome.runtime.sendMessage({
            from: 'popup',
            subject: 'setLLMId',
            id: selectedValue
        });

        await setLoadLLMStatus('loading')

        try {
            // Make a GET request to FastAPI server
            const response = await fetch(url + `changeLLM/?data=${selectedValue}`);
            const data = await response.json()
            // Replace loading icon with success message
            await setLoadLLMStatus('success')

        } catch (error) {
            console.error('Error:', error)
            // Replace loading icon with error message
            await setLoadLLMStatus('error')
            alert('Error:', error)
        }
    });
}

async function setLoadLLMStatus(status) {
    // Save the token to chrome.storage
    await chrome.runtime.sendMessage({
        from: 'popup',
        subject: 'setLLMStatus',
        status: status
    });
    const loadingIcon = document.getElementById("load-llm-icon");
    if (status && loadingIcon) {
        if (status === 'loading') {
            loadingIcon.textContent = '⏳';
        }
        if (status === 'success') {
            loadingIcon.textContent = '✅';
        }
        if (status === 'error') {
            loadingIcon.textContent = '❌';
        }
    }
}

function updateLoadLLMIcon() {
    return new Promise((resolve, reject) => {
        chrome.storage.sync.get('LLMstatus', function (result) {
            const loadingIcon = document.getElementById("load-llm-icon");
            if (result.LLMstatus && loadingIcon) {
                if (result.LLMstatus === 'loading') {
                    loadingIcon.textContent = '⏳';
                }
                if (result.LLMstatus === 'success') {
                    loadingIcon.textContent = '✅';
                }
                if (result.LLMstatus === 'error') {
                    loadingIcon.textContent = '❌';
                }
                resolve(result.LLMstatus)
            }
            console.log('No LLM loading status found');
            resolve(null); // Resolve with null if LLM status is not found
        });
    });
}

function addEventDeleteLLM() {
    document.getElementById("llm_delete_button").addEventListener("click", async function () {
        var selectedValue = document.getElementById("llm_selected").value;
        // Define default LLMs for comparison
        //const defaultLLMs = ['google/gemma-2b-it', 'stabilityai/stable-code-instruct-3b', 'codellama/CodeLlama-7b-Instruct-hf'];

        // Check if selected value is not a default LLM
        if (!defaultLLMs.includes(selectedValue)) {
            try {
                // Remove the option from the dropdown
                var select = document.getElementById("llm_selected");
                select.removeChild(select.options[select.selectedIndex]);

                // Update Chrome storage
                return new Promise((resolve, reject) => {
                    chrome.storage.sync.get({ customLlms: [] }, function (result) {
                        const filteredCustomLlms = result.customLlms.filter(llm => llm !== selectedValue);
                        chrome.storage.sync.set({ customLlms: filteredCustomLlms }, function () {
                            console.log('Custom LLM removed:', selectedValue);
                            alert('Custom LLM removed');
                        });
                    });
                    resolve(result.customLlms)
                })
            } catch (error) {
                console.error('Error:', error);
            }
        } else {
            console.log('Cannot remove a default LLM');
            alert('Cannot remove a default LLM');
        }
    });
}

function addEventSaveCustomLLM() {
    document.getElementById("custom_llm_save").addEventListener("click", async function () {
        var selectedValue = document.getElementById("custom_llm").value.trim();
        if (!selectedValue) {
            return;
        }

        try {
            chrome.storage.sync.get({ customLlms: [] }, function (result) {
                const customLlms = result.customLlms;
                // Check if the LLM ID is already saved
                if (!customLlms.includes(selectedValue)) {
                    customLlms.push(selectedValue); // Add the new LLM ID
                    chrome.storage.sync.set({ customLlms: customLlms }, function () {
                        console.log('Adding Custom LLM', selectedValue);
                        alert('Adding Custom LLM');
                        // Add the new LLM ID to the dropdown
                        const llmDropdown = document.getElementById('llm_selected');
                        const newOption = new Option(selectedValue, selectedValue, false, false);
                        llmDropdown.add(newOption, undefined);

                        // Optionally, clear the input field
                        document.getElementById("custom_llm").value = "";
                    });
                } else {
                    alert("This LLM ID is already added.");
                }
            });
        } catch (error) {
            console.error('Error:', error);
        }
    });
}




// ---------- TEXTAREA FUNCTIONS ----------

//DEPRECATED
// Add Textarea related to icon click
// async function add_LLM_Reply_Area() {
//     let parentNode = document.getElementById("partial-new-comment-form-actions");
//     if (!parentNode) {
//         console.error("Div not found for textArea");
//         return;
//     }

//     const divNode = document.createElement("div");
//     divNode.id = "popup-llm";
//     divNode.style.cssText = `
//         display: none; 
//         justify-content: space-between;
//         align-items: center;
//         margin-right: 2px;
//         width: 100%;
//         resize:none;
//     `;

//     divNode.innerHTML = `
//         <div style="flex-grow: 1;margin-right: 4px; margin-top:2px;">
//             <textarea readonly id="llm-response" class="Box" style="border: 2px solid #3b7fac; min-height:32px; height:35px; width: 100%; resize:vertical; margin-right:2px;">Waiting for LLM response ... </textarea>
//         </div>
//         <div>
//             <button type="button" class="preview_button btn-primary btn" id="copySuggestion"style="margin-right:2px;background-color:#3b7fac;">Copy</button>
//         </div>
//     `;
//     parentNode.children[0].style.width = "-webkit-fill-available";

//     parentNode.children[0].prepend(divNode);

//     //Check for reform state for copy button
//     let copyButton = document.getElementById("copySuggestion");
//     let reformState = await getToggleState("toggleReform");
//     if (copyButton && reformState != null) {
//         if (reformState === "checked") {
//             copyButton.style.display = "flex";
//         } else if (reformState === "not_checked") {
//             copyButton.style.display = "none";
//         }
//     }
// }

// Copy text
function copyToClipboard() {
    const text = document.getElementById('llm-response').value;
    if (text) {
        navigator.clipboard.writeText(text).then(() => { }).catch(error => {
            console.error('Can\'t copy text: ', error);
        });
    }
}

// Add copy button event listener
function attachCopyEvent() {
    if (document.getElementById("copySuggestion")) {
        document.addEventListener('click', function (event) {
            if (event.target.id === 'copySuggestion') {
                event.preventDefault();
                copyToClipboard();
            }
        });
    }
}

// Add text area event listener
function attachTextAreaEvent() {
    let textarea = document.getElementById("new_comment_field");
    if (textarea) {
        //EventListeners for when user gets out of textarea
        textarea.addEventListener('change', function (event) {
            updateTextArea(event.target.value);
            // if(textarea){
            //     textarea.style.display = "none";
            // }
        });
        textarea.addEventListener('blur', function (event) {
            updateTextArea(event.target.value);
            //  if(textarea){
            //     textarea.style.display = "none";
            // }
        });

        //EventListeners for when user is done writing, get the text is delayed
        textarea.addEventListener('input', getTextOvertime(function (event) {
            updateTextArea(event.target.value);
            if (event.target.value.length == 0) {
                resBox.value = "Waiting on LLM to review comment";
            }
        }), 1000);


    }
}

// Delay the get of the input
function getTextOvertime(func, waitingFunc) {
    let time;
    return function execute(...args) {
        const overtime = function () {
            clearTimeout(time);
            func(...args);
        };
        clearTimeout(time);
        time = setTimeout(overtime, waitingFunc)
    }
}

// Save with given texte
async function updateTextArea(input) {
    setCurrentComment(input);
}

// Currentcomment setter
function setCurrentComment(input) {
    currentComment = input;
}

// Currentcomment getter
function getCurrentComment() {
    return currentComment;
}

// ----------  HUGGING FACE API FUNCTIONS ---------- 

function setHuggingFaceToken(userToken) {
    // Save the token to chrome.storage
    chrome.storage.sync.set({ 'huggingFaceToken': userToken }, function () {
        console.log('Hugging Face Token is saved');
        updateHuggingFaceTokenIcon();
    });
    // Use the token immediately if needed
}

function updateHuggingFaceTokenIcon() {
    return new Promise((resolve, reject) => {
        chrome.storage.sync.get('huggingFaceToken', function (result) {
            if (result.huggingFaceToken) {
                // Token found, display green check
                document.getElementById('hugging_face_icon').textContent = '✅'; // Replace with your icon for success
                document.getElementById('hugging_face_icon').className = 'icon-success';
                resolve(result.huggingFaceToken)
            } else {
                // Token not found, display red X
                document.getElementById('hugging_face_icon').textContent = '❌'; // Replace with your icon for failure
                document.getElementById('hugging_face_icon').className = 'icon-fail';
                resolve(null)
            }
        });
    })
}


function addEventHuggingFaceSaveToken() {
    let saveButton = document.getElementById("hugging_face_token_save");
    saveButton.addEventListener("click", async () => {
        let tokenText = document.getElementById("hugging_face_token");
        if (tokenText.value) {
            setHuggingFaceToken(tokenText)
            try {
                // Make a GET request to FastAPI server
                const response = await fetch(url + `setHuggingFaceToken/?data=${tokenText.value}`);
                const data = await response.json()
            } catch (error) {
                console.error('Error:', error)
            }
        }
    });
}


// ----------  GITHUB API FUNCTIONS ---------- 

function setGitHubToken(userToken) {
    // Save the token to chrome.storage
    chrome.storage.sync.set({ 'githubToken': userToken }, function () {
        console.log('GitHub Token is saved');
        updateGitHubTokenIcon();
    });
    // Update the local variable if needed for immediate use
    token = userToken;
}

// Token getter
function getToken() {
    return token;
}

// Add event listener to watch when token is saved
function addEventGithubSaveToken() {
    let saveButton = document.getElementById("github_token_save");
    saveButton.addEventListener("click", async () => {
        let tokenText = document.getElementById("github_token");
        if (tokenText.value) {
            setGitHubToken(tokenText.value);
        }
    });
}

function updateGitHubTokenIcon() {
    chrome.storage.sync.get('githubToken', function (result) {
        if (result.githubToken) {
            document.getElementById('github_icon').textContent = '✅'; // Replace with your icon for success
            document.getElementById('github_icon').className = 'icon-success';
        } else {
            document.getElementById('github_icon').textContent = '❌'; // Replace with your icon for failure
            document.getElementById('github_icon').className = 'icon-fail';
        }
    });
}



function loadGitHubToken() {
    
    return new Promise((resolve, reject) => {
        chrome.storage.sync.get('githubToken', function (result) {
            if (result.githubToken) {
                token = result.githubToken;
                console.log('GitHub Token loaded:', result.githubToken);
                resolve(result.githubToken);
            } else {
                reject('No GitHub Token found');
            }
        });
    });
}

// Get pull request comments 
async function getPullRequestComments() {
    await loadGitHubToken()
    var headers = {
        'Authorization': `token ${token}`,
    }
    if (token) {
        try {
            var urlInfo = getInfoFromURL();
            const url = `https://api.github.com/repos/${urlInfo.owner}/${urlInfo.repo}/issues/${urlInfo.pullNumber}/comments`;
            console.log("URL:  " + url);
            const response = await fetch(url, { headers: headers });
            if (!response.ok) {
                throw new Error(`Error: ${response.status}`);
            }
            const data = await response.json();
            console.log("ResponseData:  " + data);
            const comments = data.map(async comment => {
                if (comment.user.login != "github-actions[bot]") {
                    return comment.body;
                }
            });
            return Promise.all(comments);
        } catch (error) {
            console.log(error);
        }
    } else {
        alert("PR-Comments : No Personal access token detected!")
    }
}

// Get pull request reviews 
async function getPullRequestReviews() {
    await loadGitHubToken()
    var headers = {
        'Authorization': `token ${token}`,
    }
    if (token) {
        try {
            var urlInfo = getInfoFromURL();
            const url = `https://api.github.com/repos/${urlInfo.owner}/${urlInfo.repo}/pulls/${urlInfo.pullNumber}/comments`;
            const response = await fetch(url, { headers: headers });
            if (!response.ok) {
                throw new Error(`Error: ${response.status}`);
            }
            const data = await response.json();
            const reviews = data.map(async review => {
                if (!review.in_reply_to_id) {
                    console.log("Reviews in map:" + review.body);
                    return `code en revue : ${review.diff_hunk}\ncommentaire : ${review.body}`;
                }
            });
            return Promise.all(reviews);
        } catch (error) {
            console.log(error);
        }
    } else {
        alert("PR-Reviews : No Personal access token detected!")
    }
}



// Extract info from URL to get all necessary data
function getInfoFromURL() {
    const currentUrl = window.location.href;
    const regex = /https:\/\/github\.com\/([^\/]+)\/([^\/]+)\/pull\/(\d+)/;
    const match = currentUrl.match(regex);
    if (match) {
        const owner = match[1];
        const repo = match[2];
        const pullNumber = match[3];
        return {
            owner: owner,
            repo: repo,
            pullNumber: pullNumber
        };
    } else {
        return null;
    }
}

// Get modified or added files url from Github
async function getPullRequestFiles() {
    await loadGitHubToken()
    var headers = {
        'Authorization': `token ${token}`,
    }
    if (token) {
        try {
            var urlInfo = getInfoFromURL();
            const url = `https://api.github.com/repos/${urlInfo.owner}/${urlInfo.repo}/pulls/${urlInfo.pullNumber}/files`;
            const response = await fetch(url, { headers: headers });
            if (!response.ok) {
                throw new Error(`HTTP Error: ${response.status}`);
            }
            const files = await response.json();
            return files;
        } catch (error) {
        }
    } else {
        alert("PR-FILES : No Personal access token detected!")
    }
}

// Get clean comments
function getAllPullRequestComments() {
    return getPullRequestComments().then(comments => {
        if (comments) {

            return comments;
        }
        return [];
    });
}

// Get clean comments
function getAllPullRequestReviews() {
    return getPullRequestReviews().then(reviews => {
        if (reviews) {
            console.log("getAllPullRequestReviews: " + reviews);
            return reviews;
        }
        return [];
    });
}

//Return an array of strings that contains, filename: + patch
async function getAllFileContent() {
    let files = await getPullRequestFiles();
    if (!files || files.length === 0) {
        console.log("No files found or an error occurred.");
        return [];
    }
    let filesWithPatches = files.map(file => {
        if (file.patch) {
            return `filename: ${file.filename}\npatch: ${file.patch}`;
        } else {
            return `filename: ${file.filename}\npatch: No changes or binary file`;
        }
    });
    return filesWithPatches;
}


// This function converts chrome.storage.sync.get into a promise
function getModelID() {
    return new Promise((resolve, reject) => {
        chrome.storage.sync.get("selectedLlmId", function (data) {
            if (chrome.runtime.lastError) {
                return reject(new Error(chrome.runtime.lastError));
            }
            resolve(data.selectedLlmId);
        });
    });
}

// ----------  PROMPT FUNCTION WHEN BUTTON CLICKED ---------- 
async function createPrompts() {
    //let promptsResponsesArray = [];
    let promptsResponsesMap = new Map();
    let pendingComment = getCurrentComment();
    console.log("Entering Prompt Creation...");
    console.log("PendingComments: " + pendingComment);
    if (typeof pendingComment === 'string' && pendingComment.trim() !== "") {
        let startTime = performance.now();
        console.log("Entering Prompt making");
        let promptValues = {
            generalPrompt: '',
            filePrompt: '',
            reviewPrompt: '',
            toxicityPrompt: '',
            relevancePrompt: '',
            reformPrompt: ''
        };

        // General Prompt
        const generalPromptTextarea = document.getElementById('general-prompt_prompt_area');
        if (generalPromptTextarea) { promptValues.generalPrompt = generalPromptTextarea.value; }

        // Files Prompt
        const filesTextarea = document.getElementById('files_prompt_area');
        if (filesTextarea) { promptValues.filePrompt = filesTextarea.value; }

        // Reviews Prompt
        const reviewsTextarea = document.getElementById('reviews_prompt_area');
        if (reviewsTextarea) { promptValues.reviewPrompt = reviewsTextarea.value; }

        // Toxicity Prompt
        const toxicityTextarea = document.getElementById('toxicity_prompt_area');
        if (toxicityTextarea) { promptValues.toxicityPrompt = toxicityTextarea.value; }

        // Relevance Prompt
        const relevanceTextarea = document.getElementById('relevance_prompt_area');
        if (relevanceTextarea) { promptValues.relevancePrompt = relevanceTextarea.value; }

        // Reformulation Prompt
        const reformTextarea = document.getElementById('reformulation_prompt_area');
        if (reformTextarea) { promptValues.reformPrompt = reformTextarea.value; }

        let relevanceState = await getToggleState('toggleRelevance');
        let toxicState = await getToggleState('toggleToxicity');
        let reformState = await getToggleState("toggleReform");
        let codeState = await getToggleState('toggleCode');
        let reviewState = await getToggleState('reviewCode');
        let modelID = await getModelID();
        // await chrome.storage.sync.get("selectedLlmId", function (data) {
        //     modelID = data.selectedLlmId; 
        // });
        console.log("modelID: " + modelID);

        let basePrompt = promptValues.generalPrompt;
        if (codeState === 'checked') { basePrompt += promptValues.filePrompt; }
        if (reviewState === 'checked') { basePrompt += promptValues.reviewPrompt; }

        // Relevance
        if (relevanceState === 'checked') {
            let relevanceResponse;
            let relevancePrompt = basePrompt + '\n' + promptValues.relevancePrompt;
            //console.log("Relevance Prompt: " + relevancePrompt);
            console.log("modelID Relevance: " + modelID);
            if (modelID == googleGemma2b) { relevanceResponse = await getGemmaResponse(relevancePrompt); }
            else if (modelID == stabilityAi2b) { relevanceResponse = await getStableResponse(relevancePrompt); }
            else if (modelID == tinyLlama) { relevanceResponse = await getTinyResponse(relevancePrompt); }
            else { relevanceResponse = await getDefaultLlmResponse(relevancePrompt); }
            //promptsResponsesArray.push("Relevance: " + relevanceResponse);
            promptsResponsesMap.set("Relevance", relevanceResponse);

        }
        // Toxicity
        if (toxicState === 'checked') {
            let toxicResponse;
            let toxicPrompt = basePrompt + '\n' + promptValues.toxicityPrompt;
            //console.log("Toxic Prompt:" + toxicPrompt);
            console.log("modelID Toxic: " + modelID);
            if (modelID == googleGemma2b) { toxicResponse = await getGemmaResponse(toxicPrompt); }
            else if (modelID == stabilityAi2b) { toxicResponse = await getStableResponse(toxicPrompt); }
            else if (modelID == tinyLlama) { toxicResponse = await getTinyResponse(toxicPrompt); }
            else { toxicResponse = await getDefaultLlmResponse(toxicPrompt); }
            //promptsResponsesArray.push("Toxicity: " + toxicResponse);
            promptsResponsesMap.set("Toxicity", toxicResponse);
        }
        // Reformulation
        if (reformState === 'checked') {
            let reformResponse;
            let reformPrompt = basePrompt + '\n' + promptValues.reformPrompt;
            //console.log("reform Prompt:" + reformPrompt);
            console.log("modelID Reform: " + modelID);
            if (modelID == googleGemma2b) { reformResponse = await getGemmaResponse(reformPrompt); }
            else if (modelID == stabilityAi2b) { reformResponse = await getStableResponse(reformPrompt); }
            else if (modelID == tinyLlama) { reformResponse = await getTinyResponse(reformPrompt); }
            else { reformResponse = await getDefaultLlmResponse(reformPrompt); }
            //promptsResponsesArray.push("Reformulation: " + reformResponse);
            promptsResponsesMap.set("Reformulation", reformResponse);
        }
        // Check if no prompts were toggled and add a default reply
        // if (promptsResponsesArray.length === 0) {
        //     promptsResponsesArray.push("Please toggle a prompt setting.");
        // }

        // End timer
        let endTime = performance.now();
        let timeTaken = endTime - startTime;
        //promptsResponsesArray.push(`Time taken: ${timeTaken.toFixed(2)} ms`);
    } else {
        console.log("No comment in text area");
        //promptsResponsesArray.push("Please write a comment in the text area.");
    }
    //console.log("ResponseArray: " + promptsResponsesArray);
    return promptsResponsesMap;
}

// ----------  PROMPTS GENERATION ----------

//Creates the base prompt + comments
async function getBasePrompt() {
    var basePrompt = "You are a helper bot that is assisting a programmer writing a reply to a pull request.";
    let comments = await getAllPullRequestComments();
    if (comments.length > 0) {
        basePrompt += " Here are the previous comments made on a Pull request: ";
        comments.forEach(comment => {
            basePrompt += comment;
        });
    }
    else {
        basePrompt += "No comments were made on this pull request.";
    }
    return basePrompt;
}
//
async function getFilePrompt() {
    var filePrompt = '';
    let codeState = await getToggleState('toggleCode');
    if (codeState === 'checked') {
        filePrompt += "Here are the file names and code affected by this pull request: \n";
        let fileContents = await getAllFileContent();

        // Add the filename and patch text to the prompt
        if (fileContents.length > 0) {
            fileContents.forEach(fileContent => {
                filePrompt += fileContent + "\n";
            });
        } else {
            filePrompt += "No file changes are available for this pull request.\n";
        }
    }
    return filePrompt;
}
async function getReviewPrompt() {
    var reviewPrompt = '';
    let reviewState = await getToggleState('toggleReviews');
    if (reviewState == 'checked') {
        reviewPrompt += "Here are the reviews made on this pull request: \n";
        let reviewContents = await getAllPullRequestReviews();
        // Process and add each review to the prompt
        if (reviewContents.length > 0) {
            reviewContents.forEach(reviewContent => {
                reviewPrompt += reviewContent + "\n";
            });
            reviewPrompt += "End of reviews.\n";
        } else {
            reviewPrompt += "No reviews are available for this pull request.\n";
        }
    }
    return reviewPrompt;
}

function getToxicityPrompt() {
    let pendingComment = getCurrentComment();
    let toxicityPrompt = "Here is the pending reply: " + pendingComment + '\n';
    toxicityPrompt += "Now as the helper bot, can you tell If the pending reply toxic? Your answer must be 2 sentences maximum and direct.";
    return toxicityPrompt;
}

//Returns the Relevancy/Pertinance prompt
function getRelevancePrompt() {
    let pendingComment = getCurrentComment();
    let relevancePrompt = "Here is the pending reply: " + pendingComment + '\n';
    relevancePrompt += "Now as the helper bot, can you tell If the pending reply relevant? Your answer must be 2 sentences maximum and direct.";
    return relevancePrompt;
}
//Return the reform prompt
function getReformPrompt() {
    let pendingComment = getCurrentComment();
    let reformPrompt = "Here is the pending reply: " + pendingComment + '\n';
    reformPrompt += "Now as the helper bot, Reformulate the pending reply in a professional way. Your answer must be 2 sentences maximum and direct.";
    return reformPrompt;
}

// ----------  GET LLM RESPONSE + FORMATTING ---------- 

//Function Used To call Gemma
async function getGemmaResponse(prompt) {
    const response = await fetch(url + "generate-response-Gemma", {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            prompt: prompt,
            num_tokens: 600
        })
    });
    const responseData = await response.json();
    return responseData.result.split(prompt).pop().trim().replace('<eos>', '');
}

//Function Used To call stabilityai
async function getStableResponse(prompt) {
    const response = await fetch(url + "generate-response-stable", {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            prompt: prompt,
            num_tokens: 600
        })
    });
    const responseData = await response.json();
    return responseData.result.split(prompt).pop().trim().replace('<|im_end|>', '').replace('<|endoftext|>', '');
}
async function getTinyResponse(prompt) {
    const response = await fetch(url + "generate-response-TinyLlama", {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            prompt: prompt,
            num_tokens: 600
        })
    });
    const responseData = await response.json();
    // Assuming the relevant content is after the last '</s>'
    let answer = responseData.result.split("</s>").pop().trim().replace('<|assistant|>', '');
    return answer;
}


async function getDefaultLlmResponse(prompt) {
    const response = await fetch(url + "generate-response-default", {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            prompt: prompt,
            num_tokens: 600
        })
    });
    const responseData = await response.json();
    return responseData.result;
}


// ----------  API FUNCTIONS ---------- 

// URL setter
function setURL(input) {
    url = input
}
// Check the API connexion
async function checkConnexion() {
    const response = await fetch(url + "connexion");
    return response.json();
}

// PromptID setter
async function setPromptID(id) {
    promptID = id;
}

// PromptID getter
async function getPromptID() {
    if (promptID == null) {
        await checkConnexion().then(response => {
            setPromptID(response.id);
            return response.id;
        }).catch((error) => {
            console.log(error);
        });
    }
    return promptID;
}

// Send POST request to API with prompt
async function postPrompt(data) {
    const response = await fetch(url + "generate", {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    });
    return response.json();
}
