function refresh() {
    const redis = require('redis');
    let sub = redis.createClient();
    let cnlinfos = [];
    let channel = document.getElementById("cnl");
    let info = document.getElementById("cnlinfo");

    sub.on("message", function (cnl, message) {
        ///get id
        let tmp = message.split(" ");
        let id = tmp.pop();
        cnlinfos.push(message + '\n');
        if (cnlinfos.length >= 10) {
            cnlinfos.shift();
        }
        if (tmp.pop() == "ID") {
            client.hgetall(id, function (err, obj) {
                cnlinfos.push(obj.toString() + '\n');
                info.value = cnlinfos;
            });
        } else {
            info.value = cnlinfos;
        }
        return;
    });
    sub.subscribe(channel.value);
}