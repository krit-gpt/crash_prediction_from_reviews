{\rtf1\ansi\ansicpg1252\cocoartf1348\cocoasubrtf170
{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
\paperw11900\paperh16840\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural

\f0\fs24 \cf0 ## README\
\
These	data	consist of	5,330	consumer	complaints	\
submitted	to	the	NTHSA	for	some	Honda	makes	in	years	2001-2003.\
\
Problem	was	to	build	and	validate	the	best	model	for	predicting	the	\
probability	of	a	crash	based	upon	the	topic	and	sentiment	model	and	\
upon	the	other	data	available	in	the	project	file.\
\
This	involved	the	following:\
1. Built	a	Topic	Model	that	organizes	these	complaints	into	7	groups.\
2. Scored	the	Sentiment	for	each	complaint.\
3. Merged	the	topic	group	information	and	sentiments	back	into	the	original	data	file.\
4. Built	the	best	decision	tree	to	predict	the	probability	of	a	crash.\
5. Downloaded	the	latest	news	on	the	Japanese	airbag	manufacturer	\
\'93Takata\'94	from	API and 	commented	on	how	these	articles	do	or	do	not	relate	to	the	Topic	groups found in the earlier part.}