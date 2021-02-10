//imports
var express = require("express"),
    app = express(),
    port = 8000,
    views = __dirname + "/views/";

app.set('view engine', 'ejs');

app.use(express.static(__dirname + "public"));

//renders the default page
app.get("/", function (req, res) {
    res.render("index.ejs");
});


//listens to the port
app.listen(port, function () {
    console.log("Project Has Started");
});