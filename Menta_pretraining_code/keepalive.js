function KeepAlive() {
  console.log("Keeping Colab alive...")
  document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click();
}
setInterval(KeepAlive, 60000);
