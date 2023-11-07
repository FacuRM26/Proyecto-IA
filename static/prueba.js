document.addEventListener("DOMContentLoaded", function () {
    var socket = io.connect('https://' + document.domain + ':' + location.port);

    socket.on('respuesta', function(respuesta) {
        mostrarMensajeChatbot(respuesta);
    });

    function mostrarMensajeUsuario(mensaje) {
        var chatContainer = document.getElementById("chat-container");
        var mensajeUsuario = document.createElement("p");
        mensajeUsuario.className = "mensaje-usuario";
        mensajeUsuario.textContent = mensaje;
        chatContainer.appendChild(mensajeUsuario);
    }

    function mostrarMensajeChatbot(mensaje) {
        var chatContainer = document.getElementById("chat-container");
        var mensajeChatbot = document.createElement("p");
        mensajeChatbot.className = "mensaje-chatbot";
        mensajeChatbot.textContent = mensaje;
        chatContainer.appendChild(mensajeChatbot);
    }

    function enviarMensaje() {
        var mensajeUsuario = document.getElementById("mensaje-usuario").value;
        socket.emit('mensaje', mensajeUsuario);
        mostrarMensajeUsuario(mensajeUsuario);
    
        // Vaciar el campo de texto después de 1 segundo
        setTimeout(function() {
            document.getElementById("mensaje-usuario").value = "";
        }, 1000);
    }
    

    // Agregar el evento click al botón
    var enviarButton = document.getElementById("enviar-button");
    enviarButton.addEventListener("click", enviarMensaje);

    // Mensaje de bienvenida inicial
    mostrarMensajeChatbot("Hola, soy Melody. ¿Dime que sintomas tienes?");
    
});
