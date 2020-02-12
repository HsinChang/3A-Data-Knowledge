var redis = require('redis');
var XMLHttpRequest = require("xmlhttprequest").XMLHttpRequest;
var subscriber = redis.createClient();
//It's really complex to get redis running in the Web client side
subscriber.on('message', function (channel, message) {
 let tmp = message.split(" ");
 let id = tmp.pop();
 //console.log(id);
 console.log(message);
 if (tmp.pop() == "ID") {
  printInfo(id);
 }
});
subscriber.subscribe(process.argv[2]);

function printInfo(id) {
 let xhttp = new XMLHttpRequest();
 let url = "http://localhost:3000/user/book/"+ id;
 //console.log(url);
 xhttp.open("GET", url, true);
 //console.log(id);
 xhttp.onreadystatechange = function(){
  if(xhttp.readyState==4 && xhttp.status==200){
   let response = xhttp.responseText;
   console.log(response);
  }
 };
 xhttp.send();
}
