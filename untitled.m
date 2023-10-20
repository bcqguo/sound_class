clear
clc 

fs = 16e3;
classificationRate = 20;
samplesPerCapture = fs/classificationRate;

segmentDuration = 3 ;
segmentSamples = round(segmentDuration*fs);

frameDuration = 0.020;
frameSamples = round(frameDuration*fs);

hopDuration = 0.0133;
hopSamples = round(hopDuration*fs);

FFTLength = 1024;        
numBands = 50;
overlapSamples = frameSamples - hopSamples;
numSpectrumPerSpectrogram = floor((segmentSamples-frameSamples)/hopSamples) + 1;

xfe = audioFeatureExtractor( ...
    SampleRate=fs, ...
    FFTLength=FFTLength, ...
    Window=hann(frameSamples,"periodic"), ...
    OverlapLength=overlapSamples, ...
    barkSpectrum=true);
setExtractorParameters(xfe,"barkSpectrum",NumBands=numBands,WindowNormalization=false);
numElementsPerSpectrogram = numSpectrumPerSpectrogram*numBands;


load('Snoring_CNN.mat')
labels = trainedNet_2.Layers(end).Classes;
NumLabels = numel(labels);

probBuffer = single(zeros([NumLabels,classificationRate/2]));
YBuffer = single(NumLabels * ones(1, classificationRate/2)); 

countThreshold = ceil(classificationRate*0.4);
probThreshold = single(.7);

adr = audioDeviceReader('SampleRate',fs,'SamplesPerFrame',samplesPerCapture,'OutputDataType','single');
audioBuffer = dsp.AsyncBuffer(fs);


matrixViewer = dsp.MatrixViewer("ColorBarLabel","Power per band (dB/Band)",...
    "XLabel","Frames",...
    "YLabel","Bark Bands", ...
    "Position",[400 100 600 250], ...
    "ColorLimits",[-4 2.6445], ...
    "AxisOrigin","Lower left corner", ...
    "Name","Snoring Detection");

timeScope = timescope("SampleRate",fs, ...
    "YLimits",[-1 1], ...
    "Position",[400 380 600 250], ...
    "Name","Snoring Detection", ...
    "TimeSpanSource","Property", ...
    "TimeSpan",1, ...
    "BufferLength",fs, ...
    "YLabel","Amplitude", ...
    "ShowGrid",true);
show(timeScope)
show(matrixViewer)

timeLimit = inf;

tic

while isVisible(timeScope) && isVisible(matrixViewer) && toc < timeLimit
    % Capture audio
    x = adr();
    write(audioBuffer,x);
    y = read(audioBuffer,fs,fs-samplesPerCapture);
    
    
    % Compute auditory features
    f = [zeros(floor((segmentSamples-size(y,1))/2),1);y;zeros(ceil((segmentSamples-size(y,1))/2),1)];
    features = extract(xfe,f);
    auditoryFeatures = log10(features + 1e-6);
    size(auditoryFeatures)
    
    % Perform prediction
    probs = predict(trainedNet_2, auditoryFeatures);      
    [~, YPredicted] = max(probs);
    
    % Perform statistical post processing
    YBuffer = [YBuffer(2:end),YPredicted];
    probBuffer = [probBuffer(:,2:end),probs(:)];

    [YModeIdx, count] = mode(YBuffer);
    maxProb = max(probBuffer(YModeIdx,:));
%consistent patterns to find wav form. 
%Use recordings with buffer and time delays 
    speechCommandIdx = YModeIdx;

    matrixViewer(auditoryFeatures');
    timeScope(x);

    timeScope.Title = char(labels(speechCommandIdx));

    drawnow limitrate 
end

hide(matrixViewer)
hide(timeScope)

generateMATLABFunction(afe,'extractSpeechFeatures')


