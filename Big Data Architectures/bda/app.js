"use strict"
const express = require('express');
const exphbs = require('express-handlebars');
const path = require('path');
const bodyParser = require('body-parser');
const methodOverride = require('method-override');
const redis = require('redis');

// Create Redis Client
let client = redis.createClient();
// Get expire message
client.send_command('config', ['set','notify-keyspace-events','Ex'], SubscribeExpired)
function SubscribeExpired(e,r){
  let sub = redis.createClient();
  const expired_subKey = '__keyevent@0__:expired';
  sub.subscribe(expired_subKey,function(){
    console.log(' [i] Subscribed to "'+expired_subKey+'" event channel : '+r);
    sub.on('message',function (chan,msg){console.log('[expired]',msg)});
  })
}
client.on('connect', function(){
  console.log('Connected to Redis...');
});

// Set Port
const port = 3000;

// Init app
const app = express();

// View Engine\
app.engine('handlebars', exphbs({defaultLayout:'main'}));
app.set('view engine', 'handlebars');

// body-parser
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({extended:false}));
app.use(express.static('views'));

// methodOverride
app.use(methodOverride('_method'));

//initialization
let channels = [];

app.get('/', function(req, res, next){
  res.render('publisher');
});

app.get('/user/sub', function (req, res, next) {
  res.render('subscriber');
});

app.get('/user/book/:id', function (req, res, next) {
  let id = req.params.id;
  //let rps = ;
  client.hgetall(id, function(err, obj){
    if(!obj){
      let rps  = "Book not exist";
      res.setHeader('Content-Type', 'application/json');
      //res.status(200).send(rps);
      //rps = rps.replace(/"/g,"\\\"");
      console.log(rps);
      res.end(rps);
    } else {
      let rpo = JSON.stringify(obj);
      let rps  = rpo.toString();
      //console.log(rps);
      res.setHeader('Content-Type', 'application/json');
      //res.status(200).send(rps);
      //rps = rps.replace(/"/g,"\\\"");
      console.log(rps);
      res.end(rps);
    }
  });

});
// Sub processing
/*
app.post('/user/sub/:channel', function(req, res, next){
  let channel = req.params.channel;
  let sub = redis.createClient();

    sub.on("message", function (cnl, message) {
        ///get id

    });
    sub.subscribe(channel);
  res.render('subinfo',{channel: channel});


});
*/
app.post('/user/borrow', function (req, res, next) {
  let id = req.body.id;

  client.hgetall(id, function(err, data){
    if(err || data == null) {
      res.render('subscriber', {info: "Book doesn't exist!"});
    } else if (data.borrowed == '1'){
      res.render('subscriber', {info: "Book already borrowed by others!"});
    } else {
      //refresh the expiry
      client.hmset(id, ['borrowed', '1', 'EX', 50], redis.print);
      client.expire(id, 100);
      res.render('subscriber', {info: "Book borrowed!"});
    }
  });

});

app.post('/user/return', function (req, res, next) {
  let id = req.body.id;

  client.hgetall(id, function(err, data){
    if(err || data == null) {
      res.render('subscriber', {info: "Book doesn't exist or you return too late!"});
    } else if (data.borrowed == '0'){
      res.render('subscriber', {info: "Why do you want to return an non-borrowed book?"});
    } else {
      client.hset(id, ['borrowed', '0'], redis.print);
      res.render('subscriber', {info: "Book returned!"});
    }
  });
});
// Publish Page
app.get('/user/pub', function(req, res, next){
  res.render('publisher');
});

// Process Publish
app.post('/user/pub', function(req, res, next){
  let id = req.body.id;
  let isbn = req.body.isbn;
  let title = req.body.title;
  let author = req.body.author;
  let number = req.body.number;
  let language = req.body.language;
  let pubyear = req.body.pubyear;
  let edition = req.body.edition;
  let keyword = req.body.keyword;

  client.hmset(id, [
    'isbn', isbn,
    'title', title,
    'author', author,
    'number', number,
    'language', language,
    'pubyear', pubyear,
    'edition', edition,
    'keyword', keyword,
    'borrowed', '0',
  ], function(err, reply){
    if(err){
      console.log(err);
    }
    console.log(reply);
    if(channels.length == 0) {
      channels.push(keyword);
      client.publish(keyword, "new channel " + keyword + " published.");
      client.publish(keyword, "new book added to channel " + keyword + " with ID " + id );
      console.log("New Channel: " + keyword);
    }else if(channels.includes(keyword)){
      client.publish(keyword, "new book added to channel " + keyword + " with ID " + id );
      console.log("New book to channel: " + keyword);
    } else {
      channels.push(keyword);
      client.publish(keyword, "new channel " + keyword + " published.");
      client.publish(keyword, "new book added to channel " + keyword + " with ID " + id );
      console.log("New Channel: " + keyword);
    }
    res.redirect('/');
  });
  client.expire(id, 100);
});
/*
app.get('/user/subinfo.js',function(req,res){
  res.sendFile(path.join(__dirname + '/views/subinfo.js'));
});
*/
// Delete book
app.delete('/user/delete/:id', function(req, res, next){
  client.del(req.params.id);
  res.redirect('/');
});

app.listen(port, function(){
  console.log('Server started on port '+port);
});
