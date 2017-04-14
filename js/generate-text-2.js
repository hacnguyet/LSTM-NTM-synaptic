var Neuron = synaptic.Neuron,
	Layer = synaptic.Layer,
	Network = synaptic.Network,
	Trainer = synaptic.Trainer,
	Architect = synaptic.Architect;

var LSTM = new Architect.LSTM(7,10,10,10,10,10,7);
var iterations = 10000;
var rate = .1;
var success = 0.95;
var input = [],
	output = [];
var trial,correct, criterion;

function read_and_train(){  
	var text = document.getElementById("textarea").value;
	var start = Date.now();
	correct = 0;
	criterion = 0;
	trial = 0;
	while((Date.now() - start) < 400000){
		trial++;
		for(var i = 0; i < (text.length - 1); i++){
			input = char_to_binary(text.charAt(i));
			output = char_to_binary(text.charAt(i + 1));
			var prediction = LSTM.activate(input);
			if(equal(prediction, output)){
				//console.log(binary_to_char(output));
				correct++;
			}else{
				LSTM.propagate(rate, output);
			}		
		}
		test();
		criterion = correct / (text.length - 1);
		console.log(trial, criterion * 100, Date.now() - start);
		correct = 0;
	}	
}	

function test(){
	var text = document.getElementById("text").value;
	for(var i = 0; i < (text.length - 1); i++){
		input = char_to_binary(text.charAt(i));
		output = fix_output(LSTM.activate(input));
	}
	for(var i = 0; i <= 30; i++){
		output = fix_output(LSTM.activate(output));
		text += binary_to_char(output);
	}
	console.log(text);
}

function char_to_binary(c){
	var arr = new Array(7);
	var c_ascii = (c.charCodeAt(0)).toString(2);
	c_ascii = new Array(8 - c_ascii.length).join('0') + c_ascii;	
	arr = c_ascii.split("");
	for(var i = 0; i < arr.length; i++)
		arr[i] = parseInt(arr[i],10);
	return arr;
}

function binary_to_char(arr){
	return String.fromCharCode(parseInt(arr.join(""),2));
}

function equal(prediction, output) {
  	for (var i in prediction)
   		if (Math.round(prediction[i]) != output[i])
      		return false;
  	return true;
};

function fix_output(arr){
	for(var i in arr)
		arr[i] = Math.round(arr[i]);
	return arr;
}