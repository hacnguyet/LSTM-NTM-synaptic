var vectorLength = 2;
var sequenceLength = 2;
var mem_width = 5;
var mem_size = 5;
var memoryTape = [];

for(var i = 0 ; i < mem_size; i++){
    var mem = [];
    for(var j = 0; j < mem_width; j++){
        mem.push(Math.random());
    }
    memoryTape.push(mem);
}
console.log("Intial memoryTape");
for(var i in memoryTape){
    console.log(memoryTape[i]);
}

var readtapeWeights = initializeTapeWeights();
console.log("Intial readtapeWeights");
console.log(readtapeWeights);

var writetapeWeight = initializeTapeWeights();
console.log("Intial writetapeWeight");
console.log(writetapeWeight);

var readVector =  (build_read(memoryTape,readtapeWeights));
console.log("Intial readVector");
console.log(readVector);

var inputLayer = new synaptic.Layer(vectorLength+readVector.length);
var hiddenLayer = new synaptic.Layer(100);
var headLayer = new synaptic.Layer(mem_width * 4 + mem_size * 2 + 6);
var outputLayer = new synaptic.Layer(vectorLength);

inputLayer.project(hiddenLayer,synaptic.Layer.connectionType.ALL_TO_ALL);
hiddenLayer.project(outputLayer,synaptic.Layer.connectionType.ALL_TO_ALL);  
hiddenLayer.project(headLayer,synaptic.Layer.connectionType.ALL_TO_ALL);
//headLayer.project(outputLayer,synaptic.Layer.connectionType.ALL_TO_ALL);

function train_ntm(){

    var learningRate = .01;
    var start = Date.now();
    /*------ TRAINING ----------*/
    for(var numOfRuns = 0; numOfRuns<10; numOfRuns++){
        var inputSequenceArray = [];
        var outputSequenceArray = [];
        for(var i = 0; i < sequenceLength; i++){
            var vectorTmp = [];
            while(vectorTmp.indexOf(1) == -1){
                vectorTmp = [];
                for(var j = 0; j < vectorLength; j++){
                    vectorTmp.push(Math.round(Math.random()));
                }
            }    
            inputSequenceArray.push(vectorTmp);
            outputSequenceArray.push([0,0]);
        }

        for(var i = 0; i < sequenceLength; i++){
            inputSequenceArray.push([0,0]);
            outputSequenceArray.push(inputSequenceArray[i]);
        }

        for(var i= 0; i < inputSequenceArray.length; i++){
            if((numOfRuns + 1) % 1000 == 0 && i==0){
                timeStep(inputSequenceArray[i],true); 
                console.log(numOfRuns+1,Date.now()-start);
            }else{
                //timeStep(inputSequenceArray[i],false);
                timeStep(inputSequenceArray[i],true); 
                console.log(numOfRuns+1,Date.now()-start);
            }          
            outputLayer.propagate(learningRate,outputSequenceArray[i]);
            headLayer.propagate(learningRate);
            hiddenLayer.propagate(learningRate);
        }    
    }
    console.log('Train complete');
}

function test_ntm(){
    
    var inputSequenceArray = [];
    var outputSequenceArray = [];
    for(var i = 0; i < sequenceLength; i++){
        var vectorTmp = [];
        while(vectorTmp.indexOf(1) == -1){
            vectorTmp = [];
            for(var j = 0; j < vectorLength; j++){
                vectorTmp.push(Math.round(Math.random()));
            }
        }    
        inputSequenceArray.push(vectorTmp);
    }

    for(var i = 0; i < sequenceLength; i++){
        inputSequenceArray.push([0,0]);
    }

    for(var i= 0; i < inputSequenceArray.length; i++){
        var res = timeStep(inputSequenceArray[i]);  
        console.log(inputSequenceArray[i],res);      
    }
}

function timeStep(input,flag){
    
    var testSequence = padInput(input);
    inputLayer.activate(readVector.concat(testSequence));
    if(flag){
        console.log('Input');
        console.log(readVector.concat(testSequence));
    }    

    var hiddenLayerIActiv= hiddenLayer.activate();
    /*--------------- READ WEIGHTINGS ------------------*/
    var readHeadInputs = headLayer.activate().slice(0,mem_width + mem_size + 3);
        var key = readHeadInputs.slice(0,mem_width);
        var beta = Math.exp(readHeadInputs[mem_width])+10;
        var gt = readHeadInputs[mem_width + 1];
        var shift = softmax(readHeadInputs.slice(mem_width + 2,mem_width + mem_size + 2));
        var gamma = softplus(readHeadInputs[mem_width + mem_size + 2]);

    var tmp = focus_by_content(memoryTape,key,beta);
    var tmp2 = focus_by_location(tmp,readtapeWeights,gt);
    var shift_convolveRes = shift_convolve(tmp2,shift);
    readtapeWeights = sharpen(shift_convolveRes,gamma);
    readVector =  build_read(memoryTape,readtapeWeights);
        
    /*--------------- WRITE WEIGHTINGS ------------------*/

    var writeHeadInputs = headLayer.activate().slice(mem_width + mem_size + 3,mem_width * 4 + mem_size * 2 + 6);
        var erase  = writeHeadInputs.slice(0,mem_width);
        var add = writeHeadInputs.slice(mem_width,mem_width * 2);
        var key = writeHeadInputs.slice(mem_width * 2,mem_width * 3);
        var beta = Math.exp(writeHeadInputs[mem_width * 3])+10;
        var gt = writeHeadInputs[mem_width * 3 + 1];
        var shift = softmax(writeHeadInputs.slice(mem_width * 3 + 2,mem_width * 3 + mem_size + 2));
        var gamma = softplus(writeHeadInputs[mem_width * 3 + mem_size + 2]);
            
    var tmp = focus_by_content(memoryTape,key,beta);
    var tmp2 = focus_by_location(tmp,writetapeWeight,gt);
    var shift_convolveRes = shift_convolve(tmp2,shift);
    console.log('beta');
    console.log(beta);
    console.log('key');
    console.log(key);
    console.log('tmp');
    console.log(tmp);
    console.log('tmp2');
    console.log(tmp2);
    console.log('shift');
    console.log(shift);
    console.log('shift_convolveRes');
    console.log(shift_convolveRes);
    writetapeWeight = sharpen(shift_convolveRes,gamma);
    memoryTape = build_write(memoryTape,writetapeWeight,erase,add);
       
    var res = outputLayer.activate();
    if(flag){
        console.log('memoryTape');
        for(var i in memoryTape){
            console.log(memoryTape[i]);
        }
        console.log('readtapeWeights');
        console.log(readtapeWeights);
        console.log('writetapeWeight');
        console.log(writetapeWeight);
        console.log('Output');
        console.log(res);
    }    
    return res;
}

function initializeTapeWeights(){

    var weight_init = [];    
    for (var i = 0; i < mem_size; i++){
        weight_init.push(Math.random());
    }
    
    return softmax(weight_init);
}

function tanh(x) {

    if(x === Infinity) {
    return 1;
    } else if(x === -Infinity) {
    return -1;
    } else {
    var y = Math.exp(2 * x);
    return (y - 1) / (y + 1);
    }
}

function build_read(tape_curr,weight_curr,flag){

    var tmp =[];
    for(var i in weight_curr){
        if(i == 0){
            tmp = numeric.mul(weight_curr[i],tape_curr[i]);
        }else{
            tmp = numeric.add(tmp,numeric.mul(weight_curr[i],tape_curr[i]));
        }    
        if(flag){
            console.log(i,weight_curr[i],tape_curr[i],numeric.mul(weight_curr[i],tape_curr[i]),tmp);
        }
    }
    return tmp;
}

function build_write(tape_curr,weight_curr,erase,add){

    var vector_1s = [];
    for(var i in erase){
        vector_1s.push(1);
    }

    var tapeErase = [];
    var tapeAdd = [];
    for(var i in tape_curr){
        tapeErase = numeric.mul(tape_curr[i],numeric.sub(vector_1s,numeric.mul(weight_curr[i],erase)));
        tapeAdd = numeric.add(erase,numeric.mul(weight_curr[i],add));
        tape_curr[i] = tapeAdd;
    }

    return tape_curr;
}

function padInput(input){

    var iterLength = vectorLength -input.length;
    for(var i =0; i < iterLength; i++ ){
            input.push(0);
    }

    return input;
}

function focus_by_location(tmp,prev,gt){
    
    return numeric.add(numeric.mul(gt,tmp),numeric.mul((1-gt),prev));
}

function focus_by_content(tape_curr,key,beta){

    var returnTape = [];
    for(var i in tape_curr){
        returnTape.push(beta * cosine_similarity(key,tape_curr[i]));
    }

    return softmax(returnTape);

}

function cosine_similarity(a,b){
    var array_ab = numeric.mul(a,b);
    var sum_ab = 0;
    for(var i in array_ab)
        sum_ab += array_ab[i];
    var sum_a = 0;
    for(var i in a)
        sum_a += Math.pow(a[i],2);
    var sum_b = 0;
    for(var i in b)
        sum_b += Math.pow(b[i],2);
    return sum_ab/(Math.sqrt(sum_a) * Math.sqrt(sum_b));
}

function shift_convolve(tape_curr,shift){

    var tmp = [];
    for(var i in tape_curr){
        var sum = 0;
        for(var j in tape_curr){
            if((i - j) >= 0){
                sum += tape_curr[j] * shift[i - j];
            }else if((i - j) < 0){
                sum += tape_curr[j] * shift[i - j + shift.length];
            }
        }
        tmp.push(sum);
    }

    return tmp;
}

function sharpen(tmpweight,gamma){

    var sum = 0;
    var weight_sharp =[];
    for(var i in tmpweight){
        sum += Math.pow(tmpweight[i],gamma);
    }
    for(var i in tmpweight){
        weight_sharp.push( Math.pow(tmpweight[i],gamma)/sum);
    }

    return weight_sharp;
}

function sigmoidSingle(t) {

    return 1/(1+Math.pow(Math.E, -t));
}

function softmax(array){

    var tmpSoft = [];
    var sum= 0;
    for(var i in array){
        sum += Math.exp(array[i]);
    }
    for(var i in array){
        tmpSoft.push(Math.exp(array[i])/sum);
    }

    return tmpSoft;
}

function softplus(x){

    return Math.log(1 + Math.exp(x)) + 1;
}

function sigmoid(array){

    var tmpSig= [];
    for(var i =0; i < array.length;i++){
        tmpSig.push(sigmoidSingle(array[i]));
    }

    return tmpSig;
}