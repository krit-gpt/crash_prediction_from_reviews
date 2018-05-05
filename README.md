## Predicting car crash from Complaint Text Reviews
The project involved predicting whether a crash occured or not, based on complaint reviews obtained from car customers.

The project involved using Topic Modeling, Sentiment Analysis, Data Preprocessing and Cleaning, Predictive Analysis using 
Cross Validation and Decision Tree. The best tree was obtained based on the best Accuracy obtained from Cross Validation.

#### Data
The	data	consist of	5,330	consumer	complaints submitted	to	the	NTHSA	for	some	Honda	makes	in	years	2001-2003.

#### Steps
Problem	was	to	build	and	validate	the	best	model	for	predicting	the	probability	of	a	crash	based	upon	the	topic	and	sentiment	model	and	
upon	the	other	data	available	in	the	project	file.

This	involved	the	following:
1. Built	a	Topic	Model	that	organized	these	complaints	into	7	groups.
2. Scored	the	Sentiment	for	each	complaint using the AFINN sentiment list.
3. Merged	the	topic	group	information	and	sentiments	back	into	the	original	data	file.
4. Built	the	best	decision	tree	to	predict	the	probability	of	a	crash, using the predicted topic number and the calculated sentiment as features.
5. Downloaded	the	latest	news	on	the	Japanese	airbag	manufacturer	
“Takata”	from	API and 	commented	on	how	these	articles	do	or	do	not	relate	to	the	Topic	groups found in the earlier part.
