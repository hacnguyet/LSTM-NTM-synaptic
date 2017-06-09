var Neuron = synaptic.Neuron,
	Layer = synaptic.Layer,
	Network = synaptic.Network,
	Trainer = synaptic.Trainer,
	Architect = synaptic.Architect;

var dictionary = "qwertyuiopasdfghjklzxcvbnm,.()'- \n";
var dic_length = dictionary.length;
var keys = {};
for (var i in dictionary){
	keys[dictionary[i]] = new Array(dic_length).fill(0);
	keys[dictionary[i]][i] = 1;
}

var LSTM = new Architect.LSTM(dic_length,50,dic_length);
var iterations = 1;
var rate = .1;
var input, output, prediction;
var trial = 0;
var start; 

function read_and_train(){  
	var text = document.getElementById("textarea").value;
	text = text.toLowerCase()
	.replace(/\[|\{/g, "(")
	.replace(/\]|\}/g, ")")
	.replace(/_/g, "-")
	.replace(/"/g, "'")
	.replace(/\:|\!|\?/g, ".")
	.replace(/;/g, ",");

	console.log(text.length);

	start = Date.now();
	var i = 0;
	for(var i = 0; i < (text.length - 1); i++){
		input = keys[text.charAt(i)];
		output = keys[text.charAt(i + 1)];
		prediction = LSTM.activate(input);
		LSTM.propagate(rate, output);
		console.log(i, Date.now() - start);
		if((i + 1) % 100 == 0){
			test();
		}	
		trial++;
	}
	console.log('done', trial, Date.now() - start);
}	

function test(){
	var text = document.getElementById("text").value;

	for(var i = 0; i < (text.length - 1); i++){
		input = keys[text.charAt(i)];
		output = fix_output(LSTM.activate(input));
	}
	for(var i = 0; i <= 30; i++){
		output = LSTM.activate(output);
		console.log(output);
		output = fix_output(output);
		text += binary_to_char(output);
	}
	console.log(text);
}

function binary_to_char(arr){
	var index = arr.indexOf(1);
	return dictionary.charAt(index);
}

function fix_output(arr){
	for(var i in arr)
		arr[i] = Math.round(arr[i]);
	return arr;
}