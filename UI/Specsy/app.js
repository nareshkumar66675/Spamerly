var tempDelta;
//var quill = new Quill('#editor', {
//  theme: 'snow'
//      });

var quill = new Quill('#editor', {
    modules: {
        toolbar: [
            [{ header: [1, 2, false] }],
            ['bold', 'italic', 'underline'],
            ['image', 'code-block']
        ]
    },
    scrollingContainer: '#scrolling-container',
    placeholder: 'Enter your text here to check spam',
    theme: 'snow'  // or 'bubble'
});

var getAllTextFromDelta = function (ops) {
    var str = '';
    ops.forEach(function (op) {
        if (op.hasOwnProperty('insert'))
            str = str.concat(op.insert);

    });
    return str;
};

quill.on('text-change', function (delta, oldDelta, source) {
    if (source === 'api') {
        //console.log("An API call triggered this change.");
    } else if (source === 'user') {
        tempDelta = delta;
        //console.log(delta);

        if (delta.ops[0].hasOwnProperty('retain') && delta.ops[1].hasOwnProperty('insert')) {
            if (delta.ops[1].insert === ".") { //Wait for period
                var text = quill.getText();
                varLastPrd = text.lastIndexOf(".");
                var prevPeriod = text.lastIndexOf(".", varLastPrd - 1);

                if (prevPeriod != -1) {
                    newTxt = text.substring(prevPeriod + 1, varLastPrd + 1);

                    $.ajax({
                        url: "http://localhost:7657/CheckSpam/" + newTxt
                    }).then(function (data) {

                        newTxt = newTxt.replace(/(?:\r\n|\r|\n)/g, '<br>');
                        formattedtext = "<mark class='highlight" + data.SpamCode + "' tabindex='0' data-toggle='tooltip' title='Disabled tooltip'>" + newTxt + "</mark>";
                        $("#spamText").append(formattedtext);
                        $('[data-toggle="tooltip"]').tooltip()
                    });

                    //$('p:empty').remove();
                    ////$("#spamText").html(newhtmlText)
                    //$("#spamText").html(text)

                } else { //For the first entered line

                    $.ajax({
                        url: "http://localhost:7657/CheckSpam/" + text
                    }).then(function (data) {

                        text = text.replace(/(?:\r\n|\r|\n)/g, '<br>');
                        formattedtext = "<mark class='highlight" + data.SpamCode + "'>" + text + "</mark>";
                        $("#spamText").html(formattedtext);
                    });
                }
            } else if (delta.ops[1].insert.length > 1) { // For Copy single line
                    newTxt = delta.ops[1].insert;
                    $.ajax({
                        url: "http://localhost:7657/CheckSpam/" + newTxt
                    }).then(function (data) {

                        newTxt = newTxt.replace(/(?:\r\n|\r|\n)/g, '<br>');
                        formattedtext = "<mark class='highlight" + data.SpamCode + "' data-toggle='popover'  data-trigger='hover' data-content='Disabled popover'>" + newTxt + "</mark>";
                        $("#spamText").append(formattedtext);
                        $('[data-toggle="tooltip"]').tooltip()
                    });
            }
        } else {
            text = getAllTextFromDelta(delta.ops);
            if (text.includes('.')) {
                $("#spamText").empty();
                lines = text.split(".");
                lines.forEach(function (line) {
                    $.ajax({
                        url: "http://localhost:7657/CheckSpam/" + line
                    }).then(function (data) {

                        line = line.replace(/(?:\r\n|\r|\n)/g, '<br>');
                        formattedtext = "<mark class='highlight" + data.SpamCode + "'>" + line + "</mark>";
                        $("#spamText").append(formattedtext);
                    });
                });
            }
           

            //text = text.replace(/(?:\r\n|\r|\n)/g, '<br>');
            //$("#spamText").html(text);
        }

    }
});


$(document).ready(function () {
    $(function () {
        $('[data-toggle="tooltip"]').tooltip()
    })
    $("#Bayesian").click(function (e) {
        //do something
        $("#dropdownMenuButton").text($(this).text());
        $("#dropdownMenuButton").val($(this).text());
        e.preventDefault();
    });
    $("#SVM").click(function (e) {
        $("#dropdownMenuButton").text($(this).text());
        $("#dropdownMenuButton").val($(this).text());
        e.preventDefault();
    });
    $("#NN").click(function (e) {
        $("#dropdownMenuButton").text($(this).text());
        $("#dropdownMenuButton").val($(this).text());
        e.preventDefault();
    });
});