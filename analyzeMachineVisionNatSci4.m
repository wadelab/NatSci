close all; % Clear the workspace, close all windows
clear all;

aStart=now; % Make a note of the start time

computeDataFromScratch=0;
EEGPath = '//wadelab_shared/Projects/machineLearning/natSciMachineVision/data/EEGProjectData/EEGData/'; % Where are the data? In here there should be folders called 'ARW' etc (Subject initials)
StimDataPath = '//wadelab_shared/Projects/machineLearning/natSciMachineVision/data/EEGProjectData/StimData/'; % Stimulus data (event timings and types)

% The stimulus data (order of presentations) and the EEG data are stored in
% paralled directories 'EEGData' and StimData' within ththis top--level dir
% We can access the directory listings from those two sub-dirs to make sure
% we have matching sets


sList={'01JG','02KF','03MR','04JG','05CW','06LJ','07RR','08PN','09TM','10EA','11RS','12VS','13CH','14EK','15MAT','16EF','17MW','18MP','19FM','20KP'}; % List of our subjects - we keep this here (rather than reading it using DIR in case we want to edit the list (for example to remove a subject with a lot of noise)
% Out
nSubs=length(sList);
subsToRun=nSubs;
if (computeDataFromScratch)
    for thisSub=1:subsToRun
        % Get the file lists for each subject
        
        fprintf('\Getting directories for subject %s : %d of %d\n',sList{thisSub},thisSub,nSubs);
        subj=sList{thisSub};
        EEGDataFiles{thisSub}=getEEGDataFiles(subj,EEGPath);
        EEGStimFiles{thisSub}=getEEGStimFiles(subj,StimDataPath);
    end
    
    
    for thisSub=1:subsToRun
        % Preprocess the data: Extract -200ms to 1300ms from each trial and
        % store in a stack.
        fprintf('Processing subject %d\n',thisSub);
        eegDataOut{thisSub}=arw_analyseSingleSubNatSciMachineVision(EEGDataFiles{thisSub});
        %   stimData(thisSub)=arw_getStimData(EEGStimFiles{thisSub});
    end
    
    
    % Get the image sequence data. Later we will have to check that these
    % things line up (image presentation times and EEG triggers).
    for thisSub=1:subsToRun
        fprintf('Getting image sequence data for subject %d\n',thisSub);
        imageDataOut{thisSub}=arw_extractImageSequenceData(EEGStimFiles{thisSub});
    end
    
    
    
    % The stack is now made. It is very big. Save it (uncompressed for speed)
    % so that we can reload it later
    tic
    save('/home/toolbox/AllEEGDataNatSciNoComp40hz.mat','eegDataOut','EEGDataFiles','EEGStimFiles','imageDataOut','-v7.3','-nocompression');
    toc
else
    disp('Loading data from disk...this takes a while.');
    tic
    load('/home/toolbox/AllEEGDataNatSciNoComp40hz.mat');
    toc
end



aEnd=now;
%%
% We have a small problem: The number of images is always 200 (in the stim
% files). But we don't always get 200 appropriate triggers.
% One good thing to do is to check timings. Both the EEG and the matlab
% stim PC have timestamps.
% We also have a second piece of information: The trigger numbers
% themselves are computed from the image indices (/10+x) so we should be
% able to recover the sequence almost perfectly from those.
% Note two things:
%: 1: The EEG computer runs 6 mins slow compared to the matlab computer
%: 2: The first EEG file will appear about 30 mins before the first matlab
% becuase they start, then setup the EEG cap...  Subsequent files should be
% closer in time


for thisSub=1:subsToRun
    % Let's explore this here
    subjectStimInfo=imageDataOut{thisSub};
    subjectEEGInfo=eegDataOut{thisSub};
    
    nScansInfo=length(subjectStimInfo);
    nScansEEG=length(subjectEEGInfo);
    if(nScansInfo ~= nScansEEG)
        error('Different numbers of EEG scans and stim files');
    end
    
    for thisExpt=1:length(subjectStimInfo)
        EEGTriggersCodesTimes(thisExpt)=arw_matchStimEEGTriggers(subjectEEGInfo(thisExpt).EEG, subjectStimInfo(thisExpt));
    end
    subjectTriggerCodes{thisSub}=EEGTriggersCodesTimes;
end






% Here....






%%
% ********************* THE PART BELOW IS A BIT OF FUN THAT PLOTS THE DATA
% ARW Now we load in classification accuracies and compare high with low
% class accuracy.
RANDOMIZE=0;
cVals=load('/wadelab_shared/Projects/machineLearning/darknet/darknet/DarknetClass.mat');
critVal=33; % Things with class accurary below this are 'low', above this are 'high'
%
chansToLookAt=[29,30,31];

crossSubMeans=[];
plotIntermediateFigs=1;
subsToRun=[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20];
for thisSub=subsToRun
    fprintf('\nRunning subject %.2d',thisSub);
    
    evenStack=[];
    oddStack=[];
    % concatenate across experiments
    % Do some noise rejection as
    thisEEG=eegDataOut{thisSub};
    nRuns=length(thisEEG);
    grandStack=[];
    thisSubjectTriggerData=subjectTriggerCodes{thisSub};
    
    for thisRun=1:nRuns
        thisSubjectRunCodes=thisSubjectTriggerData(thisRun);
        
        cleanEEG=arw_cleanEEGData(thisEEG(thisRun).EEG);
        % cleanEEG(isnan(cleanEEG))=0;
        imCodes=thisSubjectRunCodes.EEGVerifiedCode; % Because the cleaned data were computed from trigger times
        
        if (length(imCodes)>10)
            classVals=cVals.data.sortedClass(imCodes);
            
            
            % Here we put the data into groups. We have some options:
            % Sort by odd / even codes. This is the same as upright / inverted.
            % Sort by classVals - These are the classification values read
            % in from the DNN.
            % Or something else. Students have asked to run >just<
            % non-inverted and group by class. We can do this by...
            % 1: Identifying the indices of the odd codes
            % 2: Identifying the indices of the easy and hard ones.
            % Performing two intersections: odd int easy and odd int hard
            
            % Here is the bit where we decide whether it was hard or easy
   
            hardCodesInd=find(classVals<critVal);
            easyCodesInd=find(classVals>(100-critVal));
            
            
            evenCodesInd=find(mod(imCodes,2)==0);
            oddCodesInd=find(mod(imCodes,2)==1);
            
            
            hardOddInd=intersect(oddCodesInd,hardCodesInd);% Things that are both hard and odd
            easyOddInd=intersect(oddCodesInd,easyCodesInd); % Things that are both easy and odd
            hardEvenInd=intersect(evenCodesInd,hardCodesInd);% Things that are both hard and odd
            easyEvenInd=intersect(evenCodesInd,easyCodesInd); % Things that are both easy and odd
                 %
            d1=hardOddInd;
            d2=easyOddInd;
%             d1=hardEvenInd;
%             d2=easyEvenInd;
           
           
            if (RANDOMIZE)
            
              allD=[d1(:);d2(:)];
           
              randD=(randperm(length(imCodes),length(allD)));
            
              d1=randD(1:length(d1));
              d2=randD((length(d1)+1):end);
            end
            
            % oddCodes=d2(randperm(length(d2)));
            esBool=zeros(size(cleanEEG,3),1);
            osBool=esBool;
            esBool(d1)=1;
            osBool(d2)=1;
            evenStack=cat(1,evenStack,esBool(:));
            oddStack=cat(1,oddStack,osBool(:));
            
            % ZScore the cleaned data across electrodes
            mZ=nanmean(cleanEEG,1);
            mZ=repmat(mZ,[size(cleanEEG,1),1,1]);
            sZ=nanstd(cleanEEG,[],1);
            sZ=repmat(sZ,[size(cleanEEG,1),1,1]);
            zClean=(cleanEEG-mZ)./sZ;
            
            
            grandStack=cat(3,grandStack,zClean);

            %grandStack=cat(3,grandStack,cleanEEG);

            
            
            meanDatEven=zscore(squeeze(nanmedian(cleanEEG(:,:,d1),3)));
            stdDatEven=zscore(squeeze(nanstd(cleanEEG(:,:,d1),[],3)));
            meanDatOdd=zscore(squeeze(nanmedian(cleanEEG(:,:,d2),3)));
            stdDatOdd=zscore(squeeze(nanstd(cleanEEG(:,:,d2),[],3)));

            
            cmapH=hsv(64);
            
            if (plotIntermediateFigs)
                
                figure(thisSub+5);
                subplot(2,nRuns+1,(thisRun-1)*(2)+1);
                hold off;
                
                for thisChan=chansToLookAt
                    
                    he(thisChan)=shadedErrorBar(1:800',meanDatEven(1:800,thisChan),stdDatEven(1:800,thisChan));
                    he(thisChan).patch.FaceColor=cmapH(thisChan,:);
                    he(thisChan).patch.FaceAlpha=.2;
                    set(gca,'YLim',[-3 3]);
                    hold on;
                    
                    
                end
                
            end % End check on intermediate plotting
            
        end % End check on code length
        
    end % next run
    
    % Compute grand averages
    
    grandMeanDatEven=squeeze(nanmean((grandStack(:,:,find(evenStack))),3));
    grandStdDatEven=squeeze(nanstd((grandStack(:,:,find(evenStack))),[],3));
    grandMeanDatOdd=squeeze(nanmean((grandStack(:,:,find(oddStack))),3));
    grandStdDatOdd=squeeze(nanstd((grandStack(:,:,find(oddStack))),[],3));
    diffMeanDat=(grandMeanDatEven-grandMeanDatOdd);
    diffStdDat=grandStdDatEven;
    
    
    figure(thisSub+5);
    crossSubMeans=cat(3,crossSubMeans,diffMeanDat);
    
    subplot(2,nRuns+1,(nRuns*2)+1);
    hold off;
    for thisChan=chansToLookAt
        diffMeanDat(isnan(diffMeanDat))=0;
        diffStdDat(isnan(diffStdDat))=0;

        hg(thisChan)=shadedErrorBar(1:800',diffMeanDat(1:800,thisChan),diffStdDat(1:800,thisChan));
        hg(thisChan).patch.FaceColor=1-cmapH(thisChan,:);
        hg(thisChan).patch.FaceAlpha=.2;
        set(gca,'YLim',[-1 1]);
        
        hold on;
    end
    
    
end % Next sub

%%
% Compute a t statistic at each point for the channel average
% Note - we are going to do cluster correction using Dan's
% implementation of Oosterveld's JNS Methods paper.
nSubs=length(subsToRun);
rs=resample(double(crossSubMeans),1,10);
rs2=reshape(rs,[130,64,nSubs]);


sigPoints=d_doclustercorr(squeeze(mean(crossSubMeans(:,chansToLookAt,:),2))', 1, 0, .0005, 1000);
sigPoints2=d_doclustercorr(squeeze(mean(rs2(:,chansToLookAt,:),2))', 1, 0, .0005, 1000);




%%
figure(4);
pointsToPlot=1:1300;
hold off;

cms=crossSubMeans(:,chansToLookAt,:);
for thisChan=chansToLookAt
    mCMSChans=squeeze(mean(cms,2));
    
    overallMean=nanmedian(crossSubMeans,3);
    overallSEM=nanstd(crossSubMeans,[],3)./sqrt(size(crossSubMeans,3));
    hg(thisChan)=shadedErrorBar(pointsToPlot',overallMean(pointsToPlot,thisChan),overallSEM(pointsToPlot,thisChan));
    hg(thisChan).patch.FaceColor=1-cmapH(thisChan,:);
    hg(thisChan).patch.FaceAlpha=.2;
    set(gca,'YLim',[-2 2]);
    hold on
end
grid on
hbk=plot([200 200],[-7,7],'k');
set(hbk,'LineWidth',3);
hbx=plot([0 1000],[0 0],'k');

xlabel('Time (ms)')
ylabel('Response');

% ********** END PLOTTING FUN, BACK TO BUSINESS......
%*********************************************************************************
%%%
figure(5);
pointsToPlot=1:1300;
hold off;
cms=crossSubMeans(:,chansToLookAt,:);

mCMSChans=squeeze(mean(cms,2));

overallMean=nanmedian(mCMSChans,2);
overallSEM=nanstd(mCMSChans,[],2)./sqrt(size(mCMSChans,2));
hgn(thisChan)=shadedErrorBar(pointsToPlot',overallMean(pointsToPlot),overallSEM(pointsToPlot));
hgn(thisChan).patch.FaceColor=cmapH(30,:);
hgn(thisChan).patch.FaceAlpha=.6;
set(gca,'YLim',[-2 2]);
hold on
for thisPoint=1:length(sigPoints)
    sp=sigPoints{thisPoint};
    
    
    pH= plot(sp,.09,'o');
    set(pH,'MarkerFaceColor',[.3 .3 .3]);
    set(pH,'MarkerEdgeColor',[0 0 0]);
    
end



grid on
hbk=plot([200 200],[-7,7],'k');
set(hbk,'LineWidth',3);
hbx=plot([0 1000],[0 0],'k');

xlabel('Time (ms)')

title('Average of occipital channels rand');
%% Finally - plot versions of the graphs with clutercorrected sig values on top . We plot original and down-sampled data. The d/s data is better really.
figure(6);
subplot(1,1,1);
hold off;

pointsToPlot=1:size(rs2,1);
hold off;
cmsResamp=rs2(:,chansToLookAt,:);

mCMSChansResamp=squeeze(mean(cmsResamp,2));
xpos=-200:10:1099;


overallMeanR=nanmedian(mCMSChansResamp,2);
overallSEMR=nanstd(mCMSChansResamp,[],2)./sqrt(size(mCMSChansResamp,2));
hgn(thisChan)=shadedErrorBar(xpos',overallMeanR(pointsToPlot),overallSEMR(pointsToPlot));
hgn(thisChan).patch.FaceColor=cmapH(20,:);
hgn(thisChan).patch.FaceAlpha=.6;
set(gca,'YLim',[-.3 .3]);
hold on
for thisPoint=1:length(sigPoints2)
    sp=xpos(sigPoints2{thisPoint});%*10-200;
    pHResamp=plot(sp,.09,'o');
    set(pHResamp,'MarkerFaceColor',[.3 .3 .3]);
    set(pHResamp,'MarkerEdgeColor',[0 0 0]);
    
    
end



grid on
hbk=plot([0 0],[-7,7],'k');
set(hbk,'LineWidth',3);
hbx=plot([-200 1100],[0 0],'k');

xlabel('Time (ms)/10')

title('Average of occipital channels upright easy vs hard ');


%% Cluster correction
