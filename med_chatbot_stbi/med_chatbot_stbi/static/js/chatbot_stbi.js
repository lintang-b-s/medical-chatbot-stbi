const bot = "{% static 'img/bot.svg' %}"
const user = "{% static 'img/user.svg' %}"
console.log("hello...")
const submitButton = document.querySelector('#submit');
const outputElement = document.querySelector('#output');
const inputElement = document.querySelector('textarea');
const historyElement = document.querySelector('.history');
const buttonNewChatElement = document.querySelector('.new-chat');
const form = document.querySelector('form');
const whatContainer = document.querySelector('.what-container');

const chatContainer = document.querySelector('.chat-container')


function changeInput(value) {
    const inputElement = document.querySelector('textarea');
    inputElement.value = value;
}



function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}



let loadInterval


document.addEventListener("DOMContentLoaded", () => {
    const whatCanIHelp = document.querySelector('.what-can-i-help')
    const circle = document.getElementById("circle-icon");



    function loadWhatCanIHelp() {
        whatCanIHelp.innerHTML = ''
        let text = "What can I help with?"
        let index = 0

        let interval = setInterval(() => {
            if (index < text.length) {
                whatCanIHelp.innerHTML += text.charAt(index)
                index++

                circle.style.left = `${index * 18}px`;
            } else {
                clearInterval(interval);

                let op = 1.0;
                let fadeInterval = setInterval(() => {
                    if (op >= 0) {
                        circle.style.opacity = op;
                        op -= 0.2;
                    } else {
                        clearInterval(fadeInterval);
                    }
                }, 40)
                setTimeout(() => circle.remove(), 1000);
            }
        }, 20)

    }

    loadWhatCanIHelp();
})

function loader(element) {
    element.textContent = ''

    loadInterval = setInterval(() => {
        element.textContent += '.';

        if (element.textContent === '........') {
            element.textContent = '';
        }
    }, 300);
}

function typeText(element, text) {
    let index = 0
    let texts = text.split("\n")
    let textsSet = new Set(texts);
    texts = Array.from(textsSet);
    text = texts.join("\n");

    let interval = setInterval(() => {
        if (index < text.length) {
            element.innerHTML += text.charAt(index)
            index++
        } else {
            clearInterval(interval)
        }
    }, 10)
}


function generateUniqueId() {
    const timestamp = Date.now();
    const randomNumber = Math.random();
    const hexadecimalString = randomNumber.toString(16);

    return `id-${timestamp}-${hexadecimalString}`;
}


function chatStripe(isAi, value, uniqueId, context) {
    if (isAi) {
        return (
            `
            <div class="wrapper ai">
                <div class="chat">
                    <div class="profile">
                        <img 

                          src="/static/img/bot.svg"
                          alt="bot"
                        />
                    </div>
                    <div class="message" id=${uniqueId}>${value}</div>
                     <a class="context_button" href="#context-${uniqueId}" id="context-${uniqueId}-c">See Context/Reference</a>
                     
                     <div class="modal_container" id="context-${uniqueId}"  >
                        <div class="modal">
                                <h2 class="modal_title">Context/Reference</h2>
                                <p class="modal_text" id="text-context-${uniqueId}">${context} </p>
                                <a href="#context-${uniqueId}-c" class="context_closer"> </a>
                        </div>
                     </div>
                </div>
            </div>
        `
        )
    } else {
        return (
            `
            <div class="wrapper  user">
                <div class="chat">
                    
                    <div class="message" id=${uniqueId}>${value}</div>
                    <div class="profile">
                        <img 
                        
                          src="/static/img/user.svg"
                          alt="user"
                        />
                    </div>
                </div>
            </div>
        `
        )
    }

}

let chatHistory = []



const csrftoken = getCookie('csrftoken');

async function getMessage(e) {
    const isElementExist = document.querySelector('.what-container') !== null;
    if (isElementExist) {
        whatContainer.remove();
    }
    e.preventDefault();


    const data = new FormData(form);
    const uniqueIdUser = generateUniqueId()
    chatContainer.innerHTML += chatStripe(false, data.get('prompt'), uniqueIdUser, "")


    const uniqueId = generateUniqueId()
    chatContainer.innerHTML += chatStripe(true, " ", uniqueId, "")

    chatContainer.scrollTop = chatContainer.scrollHeight;

    const messageDiv = document.getElementById(uniqueId)
    const contextDiv = document.getElementById(`text-context-${uniqueId}`)

    loader(messageDiv)

    context = ""
    let body = inputElement.value
    if (chatHistory.length != 0) {
        for (let i = 0; i < chatHistory.length; i++) {
            context += `
            user: $ { chatHistory[i]['user'] }\
            n `
            context += `
            ai: $ { chatHistory[i]['ai'] }\
            n `
        }
    }


    const options = {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": csrftoken
        },
        body: JSON.stringify({
            question: body,
            chatHistory: context,
        })
    }

    chatHistory.push({
        'user': inputElement.value,
        'ai': ''
    })
    console.log("chatHistory: ", chatHistory)
    try {

        const response = await fetch("http://localhost:8000/ai/chatbot", options)


        clearInterval(loadInterval)
        messageDiv.innerHTML = " "

        let data;
        if (response.ok) {
            data = await response.json();
            const parsedData = data.chatbot_message.trim()

            typeText(messageDiv, parsedData)
            console.log("data: ", data)
            const context = data.context.trim()
            console.log("context: ", context)
            contextDiv.innerHTML = context

            chatHistory[-1]['ai'] = parsedData
        } else {
            const err = await response.text()

            messageDiv.innerHTML = "Something went wrong"
            alert(err)
        }



        outputElement.textContent = data.chatbot_message
        if (data.chatbot_message) {
            const pElement = document.createElement('div');
            pElement.textContent = inputElement.value;
            pElement.addEventListener('click', () => changeInput(pElement.textContent))
            historyElement.appendChild(pElement);
        }
        inputElement.value = '';
    } catch (error) {
        console.log(error)
    }
}



submitButton.addEventListener('click', getMessage)



function clearChat() {
    inputElement.value = '';
    chatContainer.innerHTML = '';
}




buttonNewChatElement.addEventListener('click', clearChat)