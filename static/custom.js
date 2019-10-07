function submit_message(message) {
    $.post("/send_message", { message: message}, handle_response);
//socketId: pusher.connection.socket_id
    function handle_response(data) {
        // append the bot repsonse to the div
        $('.chat-container').append(`
            <div class="chat-message col-sm-5  bot-message">
                ${data.message}
            </div>
      `).animate({ scrollTop: $('.chat-container').prop("scrollHeight") }, 500);

        // remove the loading indicator
        $("#loading").remove();
    }
}


$('#target').on('submit', function (e) {
    e.preventDefault();
    const input_message = $('#input_message').val()
    // return if the user does not enter any text
    if (!input_message) {
        return
    }

    $('.chat-container').append(`
        <div class="chat-message col-sm-5 offset-md-7 human-message">
            ${input_message}
        </div>
    `).animate({ scrollTop: $('.chat-container').prop("scrollHeight") }, 500);


    // loading
    $('.chat-container').append(`
        <div class="chat-message text-center col-sm-2  bot-message" id="loading">
            <b>...</b>
        </div>
    `).animate({ scrollTop: $('.chat-container').prop("scrollHeight") }, 500);


    // clear the text input
    $('#input_message').val('')

    // send the message
    submit_message(input_message)
});

// Initialize Pusher
const pusher = new Pusher('fbf2c1142db8a54b3a20', {
    cluster: 'ap2',
    encrypted: true
});

// Subscribe to movie_bot channel
const channel = pusher.subscribe('movie_bot');

// bind new_message event to movie_bot channel
channel.bind('new_message', function (data) {

    // Append human message
    $('.chat-container').append(`
            ${data.human_message}
        </div>
    `).animate({ scrollTop: $('.chat-container').prop("scrollHeight") }, 500);


    // Append bot message
    $('.chat-container').append(`
        <div class="chat-message col-sm-5  bot-message">
            ${data.bot_message}
        </div>
    `).animate({ scrollTop: $('.chat-container').prop("scrollHeight") }, 500);

})