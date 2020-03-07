
function getWords()
{
   glove();

    
}

function glove()
{
    var left_grid = document.getElementById("grid-left");

    var word = document.getElementById("word").value;
    var amount = document.getElementById("drop_amount").value;
    var dimension = document.getElementById("dimension").value;
    w2v();
    //stuur get request naar flask server om de eerste x-aantal woorden terug te krijgen die het beste bij het meegegeven woord passen voor Glove
    $.ajax({
        url: "http://127.0.0.1:5000/glove",
        type: "GET",
        data: {'word': word , 'amount': amount, 'dimension': dimension},
        success: function(response){
            var child = left_grid.firstChild;

            while(child){
                left_grid.removeChild(child);
                child = left_grid.firstChild;
            }

            var result = response;
            result = result.substr(1);
            result = result.slice(0, -1);
            
            var parsed ="";

            for (var i = 0; i != result.length + 1; i++){
       
                if (result.charAt(i) == '(' || result.charAt(i) == '\'' || result.charAt(i) == ')' || result.charAt(i) == ','){
             }
             else{
                 parsed += result.charAt(i);
             } 
         }
         
            output_array = parsed.split(" ")
    
            var output = ""
            var ul = document.createElement("ul");
            left_grid.appendChild(ul);            
            
            for (var i = 0; i != output_array.length; i++){
                    
                if (i%2 == 0){
                    var li = document.createElement("li");
                    var text = document.createTextNode(output_array[i] + ": " + output_array[i + 1]);
                    li.appendChild(text);
                    
                    ul.appendChild(li);            
                }
                else{
                }            
            }

           
        },
        error: function(){
            alert('Meegegeven woord bestaat niet!');
          }
        
           
    });

   
}

function w2v()
{
    var right_grid = document.getElementById("grid-right");

    var word = document.getElementById("word").value;
    var amount = document.getElementById("drop_amount").value;
    var dimension = document.getElementById("dimension").value;
    //stuur get request naar flask server om de eerste x-aantal woorden terug te krijgen die het beste bij het meegegeven woord passen voor Word2Vec (fasttext)
    $.ajax({
        url: "http://127.0.0.1:5000/w2v",
        type: "GET",
        data: {'word': word , 'amount': amount, 'dimension': dimension},
        success: function(response){
            console.log(response)
            var child = right_grid.firstChild;

            while(child){
                right_grid.removeChild(child);
                child = right_grid.firstChild;
            }

            var result = response;
            result = result.substr(1);
            result = result.slice(0, -1);
            
            var parsed ="";

            for (var i = 0; i != result.length + 1; i++){
       
                if (result.charAt(i) == '(' || result.charAt(i) == '\'' || result.charAt(i) == ')' || result.charAt(i) == ','){
             }
             else{
                 parsed += result.charAt(i);
             } 
         }
         
            output_array = parsed.split(" ")
    
            var output = ""
            var ul = document.createElement("ul");
            right_grid.appendChild(ul);            
            
            for (var i = 0; i != output_array.length; i++){
                   
                if (i%2 == 0){
                    var li = document.createElement("li");
                    var text = document.createTextNode(output_array[i + 1] + ": " + output_array[i]);
                    li.appendChild(text);
                    
                    ul.appendChild(li);            
                }
                else{
                }            
            }
           
        },
        error: function(){
            alert('error!');
          }
        
           
    });
}

function similarity()
{
    var gs = document.getElementById('gs');

    var w1 = document.getElementById('w1').value;
    var w2 = document.getElementById('w2').value;
    similarityW2V();
    //stuur get request naar flask server om de cosinusafstand terug te krijgen voor de 2 meegegeven woorden voor Glove
    $.ajax({
        url: "http://127.0.0.1:5000/similarityGlove",
        type: "GET",
        data: {'w1': w1 , 'w2': w2},
        success: function(response){

            var child = gs.firstChild;

            while(child){
                gs.removeChild(child);
                child = gs.firstChild;
            }
            var p = document.createElement('p');

            var result = document.createTextNode("Glove: " + response);
            p.appendChild(result);
            gs.appendChild(p);
            
        },
        error: function(){
            alert('error!');
          }

    });

}

function similarityW2V()
{

    var w2v = document.getElementById('w2v');

    var w1 = document.getElementById('w1').value;
    var w2 = document.getElementById('w2').value;
    
    //stuur get request naar flask server om de cosinusafstand terug te krijgen voor de 2 meegegeven woorden voor Word2Vec (fasttext)
    $.ajax({
        url: "http://127.0.0.1:5000/similarityW2V",
        type: "GET",
        data: {'w1': w1 , 'w2': w2},
        success: function(response){

            var child = w2v.firstChild;

            while(child){
                w2v.removeChild(child);
                child = w2v.firstChild;
            }
            var p = document.createElement('p');

            var result = document.createTextNode("Word2Vec: " + response);
            p.appendChild(result);
            w2v.appendChild(p);
            //console.log(response);
        },
        error: function(){
            alert('error!');
          }

    });
}

function predictNextWords()
{
    predicted_div = document.getElementById('predicted_words');
    sentence = document.getElementById('sentence').value;
    amount = document.getElementById('amount_predict').value;

    if(sentence == "")
    {
        alert('please give a sentence of at least 1 word');
    }

    else
    {
        if( parseInt(amount) <= 0)
        {
            alert('give a number greater than 0');
        }

        else
        {
            //stuur get request naar flask server om het x-aantal volgende woorden te voorspellen voor de meegegeven text
            $.ajax({
                url: "http://127.0.0.1:5000/predictNWords",
                type: "GET",
                data: {'tekst': sentence, 'aantal': amount},
                success: function(response){
                    
                    var child = predicted_div.firstChild;

                    while(child){
                        predicted_div.removeChild(child);
                        child = predicted_div.firstChild;
                    }

                    p = document.createElement('p');
                    result = document.createTextNode("Result: " + response);
                    p.appendChild(result);
                    predicted_div.appendChild(p);
                    
        
                },
                error: function(){
                    alert('error!');
                }
            })
        }
    }

}

function gap_fill()
{
    gap_div = document.getElementById('gap_word');
    zin = document.getElementById('zin').value;
    w1 = document.getElementById('pw1').value;
    w2 = document.getElementById('pw2').value;
    w3 = document.getElementById('pw3').value;
    //stuur get request naar flask server om de beste match te vinden tussen de 3 meegegeven woorden om in de zin te plaatsen voor
    $.ajax({
        url: "http://127.0.0.1:5000/gap",
        type: "GET",
        data: {'w1': w1, 'w2': w2, 'w3': w3, 'zin': zin},
        success: function(response){

            var child = gap_div.firstChild;

            while(child){
                gap_div.removeChild(child);
                child = gap_div.firstChild;
            }

            p = document.createElement('p');
            result = document.createTextNode(response);
            p.appendChild(result);
            gap_div.appendChild(p);
        },
        error: function(){
            alert('error!');
        }
    })
}