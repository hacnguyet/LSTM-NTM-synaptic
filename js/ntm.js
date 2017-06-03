var vectorLength = 4;
var sequenceLength = 2;
var shiftSize = 3;
var mem_width = 5;
var mem_size = 10;
var learningRate = .1;
var iterations = 1000;
var log = 1000;

var memoryTape = [];
var memoryTape_tmp = [];
var readtapeWeight = [];
var readtapeWeight_tmp = [];
var writetapeWeight = [];
var writetapeWeight_tmp = [];
var readVector = [];
var headLayerActivate = [];
var lastHiddenLayerActivate = [];
var memoryGradient = [];

var inputLayer = new synaptic.Layer(vectorLength);
var readLayer = new synaptic.Layer(mem_width);
var hiddenLayer = new synaptic.Layer(100);
var headLayer = new synaptic.Layer(mem_width * 4 + shiftSize * 2 + 6);
var outputLayer = new synaptic.Layer(vectorLength);

inputLayer.project(hiddenLayer,synaptic.Layer.connectionType.ALL_TO_ALL);
readLayer.project(hiddenLayer,synaptic.Layer.connectionType.ALL_TO_ALL);
hiddenLayer.project(outputLayer,synaptic.Layer.connectionType.ALL_TO_ALL);
hiddenLayer.project(headLayer,synaptic.Layer.connectionType.ALL_TO_ALL);

function train_ntm(){

    var start = Date.now();
    /*------ TRAINING ----------*/
    console.log('Train start');

    for(var numOfRuns = 0; numOfRuns < iterations; numOfRuns++){
        //reset memoryMatrix, readWeight, writeWeight back to init 
        useModel();

        //init train sequence
        var sequence = initSequence(2);
        var inputSequenceArray = sequence[0];
        var outputSequenceArray = sequence[1];

        //train sequence
        for(var i in inputSequenceArray){
            forwardController(inputSequenceArray[i],false);      
            outputLayer.propagate(learningRate,outputSequenceArray[i]);
            hiddenLayer.propagate(learningRate);
            readLayer.propagate(learningRate);
            if(i > 0){ 
               propagateMemory(outputSequenceArray[i].concat([0]));
               //propagateMemory(readLayer.activate());
            }    
            timeStep(false);
            if((numOfRuns + 1) % 1 == 0){
                //console.log(numOfRuns+1,readVector);
            }
        }  

        if((numOfRuns + 1) % log == 0){
            console.log('==============',numOfRuns+1,Date.now()-start,'==============');
            test_ntm(2);
        } 
    }
    console.log('Train complete');
}

function test_ntm(n){
    //Init memoryMatrix, readWeight, writeWeight
    useModel();

    //init test sequence
    var sequence = initSequence(1);
    if(n){
        var sequence = initSequence(n);
    }
    var inputSequenceArray = sequence[0];
    var outputSequenceArray = sequence[1];

    //test
    drawSequence('memoryTape',memoryTape);
    for(var i in inputSequenceArray){
        forwardController(inputSequenceArray[i],true);      
        timeStep(true);
    }  

    //draw
    drawSequence('memoryTape',memoryTape);
    drawSequence('output',outputSequenceArray);

}

function forwardController(input,flag){

    inputLayer.activate(input);
    readLayer.activate(readVector);
    if(flag){
        drawSequence('input',[readVector.concat(input)]);
    }    

    hiddenLayerActivate = hiddenLayer.activate();
    var outputLayerActivate = outputLayer.activate();
    if(flag){
        drawSequence('prediction',[outputLayerActivate]);
    }    

    return outputLayerActivate;
}

function timeStep(flag){

    //Save machine state
    lastHiddenLayerActivate = hiddenLayerActivate.slice();
    hiddenLayer.activate(hiddenLayerActivate);
    headLayerActivate = headLayer.activate();
    // console.log('time_hiddenLayerActivate',hiddenLayerActivate);
    //console.log('time_headLayerActivate',headLayerActivate);

    //Compute next machine state
    var nextMachineState = memoryMechanism(headLayerActivate,memoryTape,writetapeWeight,readtapeWeight,flag);
    computeMemoryGradient();
    readVector = nextMachineState[0];
    memoryTape = nextMachineState[1];
    writetapeWeight = nextMachineState[2];
    readtapeWeight = nextMachineState[3];

}

function memoryMechanism(prevHeadLayerActivate,prevMemoryTape,prevWritetapeWeight,prevReadtapeWeight,flag){

    /*--------------- WRITE WEIGHTINGS ------------------*/

    var writeHeadInputs = prevHeadLayerActivate.slice(0,mem_width * 3 + shiftSize + 3);
        var erase  = writeHeadInputs.slice(0,mem_width);
        var add = writeHeadInputs.slice(mem_width,mem_width * 2);
        var key = writeHeadInputs.slice(mem_width * 2,mem_width * 3);
        var beta = Math.exp(writeHeadInputs[mem_width * 3]);
        var gt = writeHeadInputs[mem_width * 3 + 1];
        var shift = softmax(writeHeadInputs.slice(mem_width * 3 + 2,mem_width * 3 + shiftSize + 2));
        if(flag){
            drawSequence('writeShift',[shift]);
        }
        shift = padShift(shift);
        var gamma = softplus(writeHeadInputs[mem_width * 3 + shiftSize + 2]) + 1;
            
    var tmp = focus_by_content(prevMemoryTape,key,beta);
    var tmp2 = focus_by_location(tmp,prevWritetapeWeight,gt);
    var shift_convolveRes = shift_convolve(tmp2,shift);
    var nextWritetapeWeight = sharpen(shift_convolveRes,gamma);
    var nextMemoryTape = build_write(prevMemoryTape,nextWritetapeWeight,erase,add);
    if(flag){
            drawSequence('erase',[erase]);
            drawSequence('add',[add]);
            drawSequence('writeKey',[key]);
            drawSequence('writeTmp',[tmp]);
            drawSequence('writeTmp2',[tmp2]);
            drawSequence('writeShiftConv',[shift_convolveRes]);
            drawSequence('writetapeWeight',[nextWritetapeWeight]);
    }

    /*--------------- READ WEIGHTINGS ------------------*/

    var readHeadInputs = prevHeadLayerActivate.slice(mem_width * 3 + shiftSize + 3,mem_width * 4 + shiftSize * 2 + 6);
        var key = readHeadInputs.slice(0,mem_width);
        var beta = Math.exp(readHeadInputs[mem_width]);
        var gt = readHeadInputs[mem_width + 1];
        var shift = softmax(readHeadInputs.slice(mem_width + 2,mem_width + shiftSize + 2));
        if(flag){
            drawSequence('readShift',[shift]);
        }
        shift = padShift(shift);
        var gamma = softplus(readHeadInputs[mem_width + shiftSize + 2]) + 1;

    var tmp = focus_by_content(nextMemoryTape,key,beta);
    var tmp2 = focus_by_location(tmp,prevReadtapeWeight,gt);
    var shift_convolveRes = shift_convolve(tmp2,shift);
    var nextReadtapeWeight = sharpen(shift_convolveRes,gamma);
    var nextReadVector =  build_read(nextMemoryTape,nextReadtapeWeight);
    if(flag){
        drawSequence('readKey',[key]);
        drawSequence('readTmp',[tmp]);
        drawSequence('readTmp2',[tmp2]);
        drawSequence('readShiftConv',[shift_convolveRes]);
        drawSequence('readVector',[nextReadVector]);
        drawSequence('readtapeWeight',[nextReadtapeWeight]);
    }    

    return [nextReadVector, nextMemoryTape, nextWritetapeWeight, nextReadtapeWeight];

}

function computeMemoryGradient(){

    var nextState = memoryMechanism(headLayerActivate,memoryTape,writetapeWeight,readtapeWeight,false);
    var nextReadVector = nextState[0];

    memoryGradient = [];
    for(var i in headLayerActivate){
        headLayerActivate[i] += 0.0001;
        var nextState_tmp = memoryMechanism(headLayerActivate,memoryTape,writetapeWeight,readtapeWeight,false);
        var nextReadVector_tmp = nextState_tmp[0];
        var memoryGradient_tmp = [];
        for(var j in nextReadVector_tmp){
            memoryGradient_tmp.push((nextReadVector_tmp[j] - nextReadVector[j]) / 0.0001);
        }
        memoryGradient.push(memoryGradient_tmp);
        headLayerActivate[i] -= 0.0001;
    }
    
}

function propagateMemory(readVector_tmp){
    
    hiddenLayer.activate(lastHiddenLayerActivate);
    var lastHeadLayerActivate = headLayer.activate();

    for(var i in readVector){
        for(var j in memoryGradient){
            lastHeadLayerActivate[j] -= learningRate * memoryGradient[j][i] * (readVector[i] - readVector_tmp[i])
        }
    } 

    headLayer.propagate(learningRate, lastHeadLayerActivate);

}

function initSequence(n){

    var inputSequenceArray = [];
    var outputSequenceArray = [];
    var vectorZero = [];
    for(var j = 0; j < vectorLength; j++){
        vectorZero.push(0);
    }

    switch(n){
        case 1:
            var inputSequenceArray = [[0,0,1,0],[0,0,1,1],[0,0,0,0],[0,0,0,0]];
            var outputSequenceArray = [[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,1,1]];
            break;
        case 2:
            //add vector flag start
            var vectorStart = [];
            for(var j = 0; j < vectorLength; j++){
                vectorStart.push(0);
            }
            vectorStart[vectorLength - 2] = 1;
            inputSequenceArray.push(vectorStart);
            outputSequenceArray.push(vectorZero);

            //add random vector  
            for(var i = 0; i < sequenceLength; i++){
                var vectorRandom = [];
                while(vectorRandom.indexOf(1) == -1){
                    vectorRandom = [];
                    for(var j = 0; j < vectorLength - 2; j++){
                        vectorRandom.push(Math.round(Math.random()));
                    }    
                }  
                vectorRandom.push(0);
                vectorRandom.push(0);  
                inputSequenceArray.push(vectorRandom);
                outputSequenceArray.push(vectorZero);
            }

            //add vector flag end
            var vectorEnd = [];
            for(var j = 0; j < vectorLength; j++){
                vectorEnd.push(0);
            }
            vectorEnd[vectorLength - 1] = 1;
            inputSequenceArray.push(vectorEnd);
            outputSequenceArray.push(vectorZero);

            for(var i = 1; i < sequenceLength + 1; i++){
                inputSequenceArray.push(vectorZero);
                outputSequenceArray.push(inputSequenceArray[i]);
            }       
            break;
    }
    
    return [inputSequenceArray,outputSequenceArray];
}

function initModel(){
    memoryTape_tmp = [];
    for(var i = 0 ; i < mem_size; i++){
        var mem = [];
        for(var j = 0; j < mem_width; j++){
            mem.push(Math.random());
        }
        memoryTape_tmp.push(mem);
    }

    readtapeWeight_tmp = initializeTapeWeights(true);
    writetapeWeight_tmp = initializeTapeWeights(true);
    console.log('Initialized Model');

}

function initializeTapeWeights(flag){

    var weight_init = [];   
    for (var i = 0; i < mem_size; i++){
        if(flag){       
            weight_init.push(Math.random());
        }else{
            weight_init.push(0.00001);
        }    
    }
    console.log(softmax(weight_init));
    return softmax(weight_init);
}

function useModel(){

    memoryTape = [];
    for(var i in memoryTape_tmp){
        var tmp = [];
        for(var j in memoryTape_tmp[i]){
            tmp.push(memoryTape_tmp[i][j])
        }
        memoryTape.push(tmp);
    }

    writetapeWeight = writetapeWeight_tmp.slice();
    readtapeWeight = readtapeWeight_tmp.slice();
    readVector = build_read(memoryTape_tmp,readtapeWeight_tmp);
    
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
    var tapeNext = [];
    for(var i in tape_curr){
        tapeErase = numeric.mul(tape_curr[i],numeric.sub(vector_1s,numeric.mul(weight_curr[i],erase)));
        tapeAdd = numeric.add(tapeErase,numeric.mul(weight_curr[i],add));
        tapeNext.push(tapeAdd);
    }

    return tapeNext;
}

function padShift(shift){

    if(shift.length <= 2){
        return shift;
    }

    var iterLength = mem_size - shift.length;
    var tmp = shift.slice(1,shift.length);
    for(var i = 0; i < iterLength; i++ ){
            tmp.push(0);
    }
    tmp.push(shift[0]);

    return tmp;
}

function focus_by_content(tape_curr,key,beta){

    var returnTape = [];
    for(var i in tape_curr){
        returnTape.push(beta * cosine_similarity(key,tape_curr[i]));
    }

    return softmax(returnTape);

}

function focus_by_location(tmp,prev,gt){
    
    return numeric.add(numeric.mul(gt,tmp),numeric.mul((1-gt),prev));
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
    return sum_ab/(Math.sqrt(sum_a) * Math.sqrt(sum_b) + 1e-5);
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

    return Math.log(1 + Math.exp(x));
}

function sigmoid(array){

    var tmpSig= [];
    for(var i =0; i < array.length;i++){
        tmpSig.push(sigmoidSingle(array[i]));
    }

    return tmpSig;
}

function rectifier(array){

    for(var i in array){
        array[i] = rectifierSingle(array[i]);
    }

    return array;
}

function rectifierSingle(i){

    if(i < 0)
        i = 0;

    return i;
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

function setSquash(layer, squash){

    for(var i in layer.list){
        layer.list[i].squash = squash;
    }

    return layer;
}
