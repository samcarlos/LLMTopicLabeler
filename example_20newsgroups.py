import numpy as np
import pandas as pd
import ollama 

from sklearn.datasets import fetch_20newsgroups
from LLMTopicLabeler import LLMTopicLabeler

#load data
newsgroups_train = fetch_20newsgroups(subset='train')

def embed_topic_text(topic_text: str) -> np.array:
    # Use ollama to embed the topic text
    response = ollama.embeddings(model="mxbai-embed-large", prompt=topic_text)
    temp_embedding = np.array(response["embedding"]).reshape(1,-1)
    return temp_embedding

#embed and append raw text to df
embeddings = [embed_topic_text(x) for x in newsgroups_train.data]
embeddings_pd = pd.DataFrame(np.array(embeddings)[:,0,:])
embeddings_pd.columns = ['embedding_' + str(x) for x in range(1024)]
embeddings_pd['paragraph'] = newsgroups_train.data

#build model
auto_classifier = LLMTopicLabeler()
auto_classifier.iterative_topic_classification('taxes', embeddings_pd, 
                                                                                            y_iterations = 5, quantiles_cutoff = [.999, .995,.99,.985])
#predict
predictions = auto_classifier.predict(embeddings_pd)

#print the top five predicted texts of the class
for x in np.where(predictions == 1)[0][:5]:
    print(newsgroups_train.data[x])
    print('\n')
    print('_______')




##output 

From: arf@genesis.MCS.COM (Jack Schmidling)
Subject: Re: The U.S. Holocaust Memorial Museum: A Costly and Dangerous Mistake
Organization: MCSNet Contributor, Chicago, IL
Lines: 19
NNTP-Posting-Host: localhost.mcs.com

In article <C5ut1s.3xA@bony1.bony.com> jake@bony1.bony.com (Jake Livni) writes:
>through private contributions on Federal land".  Your hate-mongering
>article is devoid of current and historical fact, intellectual content
>and social value.  Down the toilet it goes.....
>

And we all know what an unbiased source the NYT is when it comes to things
concerning Israel.

Neither the Times nor the trained seals who have responded thus far seem to
recognize the statement that these "private funds" were all tax exmpt.  In
otherwords, American taxpayers put up at least 30% of the money.  And
finalyy, how does "Federal land" mitigate the offensiveness of this alien
monument dedicated to perpetuating pitty and the continual flow of tax money
to a foreign entity?

That "Federal land" and tax money could have been used to commerate
Americans or better yet, to house homeless Americans.




_______
From: ohayon@jcpltyo.JCPL.CO.JP (Tsiel Ohayon)
Subject: Re: The U.S. Holocaust Memorial Museum: A Costly and Dangerous Mistake
Organization: James Capel Pacific Limited, Tokyo Japan
Lines: 33

In article <C5ut1s.3xA@bony1.bony.com> jake@bony1.bony.com (Jake Livni) writes:
   In article <1r3n8d$4m5@techbook.techbook.com> Dan Gannon writes:

[DG] THE U.S. HOLOCAUST MEMORIAL MUSEUM: A COSTLY AND DANGEROUS MISTAKE
[DG] by Theodore J. O'Keefe
[DG] HARD BY THE WASHINGTON MONUMENT, within clear view of the Jefferson
[DG] Memorial, an easy stroll down the Mall to the majestic Lincoln Memorial,
[DG] has arisen, on some of the most hallowed territory of the United States of
[DG] America, a costly and dangerous mistake.  On ground where no monument yet
[DG] marks countless sacrifices and unheralded achievements of Americans of all
[DG] races and creeds in the building and defense of this nation, sits today a
[DG] massive and costly edifice, devoted above all to a contentious and false
[DG] version of the ordeal in Europe during World War II, of non-American
[DG] members of a minority, sectarian group.  Now, in the deceptive guise of
[DG] tolerance, the United States Holocaust Memorial Museum begins a propaganda
[DG] campaign, financed through the unwitting largess of the American taxpayer,
[DG] in the interests of Israel and its adherents in America.

[JAKE] After reading the first paragraph, a quick scan confirmed my first
[JAKE] impression: this is a bunch of revisionist and anti-semitic hogwash.

Jake, I'm really disappointed in you. It took you a whole paragraph
to see that it was "bunch of revisionist and anti-semitic hogwash". :-)

The article title "THE U.S. HOLOCAUST MEMORIAL MUSEUM: A COSTLY AND 
DANGEROUS MISTAKE" should have been enough! :-)

Tsiel
-- 
----8<--------------------------------------------------------------->8------
Tsiel:ohayon@jcpl.co.jp	   | If you do not receive this E-mail, please let me
Employer may not have same | know as soon as possible, if possible.
opinions, if any !         | Two percent of zero is almost nothing.



_______
From: VEAL@utkvm1.utk.edu (David Veal)
Subject: Re: National Sales Tax, The Movie
Lines: 66
Organization: University of Tennessee Division of Continuing Education

In article <1993Apr16.164750.21913@alchemy.chem.utoronto.ca> golchowy@alchemy.chem.utoronto.ca (Gerald Olchowy) writes:

>In article <9304151442.AA05233@inet-gw-2.pa.dec.com> blh@uiboise.idbsu.edu (Broward L. Horne) writes:
>>      Well, it seems the "National Sales Tax" has gotten its very
>
>>      own CNN news LOGO!
>>
>>      Cool.  That means we'll be seeing it often.
>>
>>      Man, I sure am GLAD that I quit working ( or taking this 
>>      seriously ) in 1990.  If I kept busting my ass, watching 
>>      time go by, being frustrated, I'd be pretty DAMN MAD by 
>>      now.
>>      
>>      I just wish I had the e-mail address of total gumby who
>>      was saying that " Clinton didn't propose a NST ".
>>
>
>Actually, Jerry Brown essentially did...and Clinton, in his demagogue
>persona, condemned Brown for it in the crucial NY primary last year.
>
>However....
>
>Why don't the Republicans get their act together, and say they
>will support a broad-based VAT that would have to be visible
>(the VAT in Canada is visible unlike the invisible VATS they
>have in Europe)
>and suggest a rate sufficient to halve income and corporate
>and capital gains tax rates and at a rate sufficient to give
>the Clintons enough revenue for their health care reform, 

       The Republicans are, in general, fighting any tax increase.
There is also worry that a VAT would be far too easy to increase
incrementally.

       (BTW, what is different between Canada's tax and most of
Europe's that makes it "visible?")

>and
>force an agreement with the Democrats that the top income tax
>rate would then be frozen for the forseeable future and could
>be increased only via a national referendum.

       This would require a constitutional amendment, and Congress
enjoys raising taxes too much to restrict themselves like that.
(Besides, with the 2/3 majority necessary to pull that off you'd 
have a difficult time "forcing" anything like that.)

>Why not make use of the Clintons to do something worthwhile...
>shift the tax burden from investment to consumption, and get
>health care reform, and a frozen low top marginal tax rate
>all in one fell swoop.

       Primarily because it's a practical impossibility to "freeze"
tax rates.

       However, this is something that bothers me.  We're always talking
about "consumer confidence" and "consumer spending" as gauges for the
economy.  If they really are important, wouldn't shifting taxes to
consumption provide a disincentive to spend money?

------------------------------------------------------------------------
David Veal Univ. of Tenn. Div. of Cont. Education Info. Services Group
PA146008@utkvm1.utk.edu - "I still remember the way you laughed, the day
your pushed me down the elevator shaft;  I'm beginning to think you don't
love me anymore." - "Weird Al"



_______
From: ipser@solomon.technet.sg (Ed Ipser)
Subject: Re: Supply Side Economic Policy (was Re: David Stockman )
Nntp-Posting-Host: solomon.technet.sg
Organization: TECHNET, Singapore
Distribution: na
Lines: 29

In article <Ufk_Gqu00WBKE7cX5V@andrew.cmu.edu> Ashish Arora <ashish+@andrew.cmu.edu> writes:
>Excerpts from netnews.sci.econ: 5-Apr-93 Re: Supply Side Economic Po..
>by Not a Boomer@desire.wrig 
>[...]
>
>>    The deficits declined from 84-9, reaching a low of 2.9% of GNP before  
>> the tax and spending hike of 1990 reversed the trend.
>>  
>> Brett
>Is this true ?  Some more details would be appreciated.

Yes, sadly, this is true. The primary reason, and the essence of the
details that you are seeking, is that the Grahm-Rudman budget controls
were working.  In fact, they were working so well that unless the feds
did something, they were going to have to start cutting pork. So Bush
and the Democrats got together in a Budget Summit and replaced
Grahm-Rudman with the now historic Grand Compromise in which Bush
"consented" to raise taxes in exchange for certain caps on spending
increases.

As it turned out, the taxes killed the Reagan expansion and the caps
on spending increases were dispelled by Clinton in his first act as
President (so that he could create his own new plan with more tax
increases).

The result is that Clinton now HOPES to reduce the deficit to a level 
ABOVE where it was when Reagan left office.

Chew on that awhile.



_______
From: bernard@sirius.gte.com (Bernard Silver)
Subject: Re: Bill Conklin (et al) 's letter
	<1993Apr3.231858.27507@midway.uchicago.edu>
Organization: GTE Laboratories Incorporated
Lines: 27
In-reply-to: thf2@kimbark.uchicago.edu's message of 3 Apr 93 23:18:58 GMT

In article <1993Apr3.231858.27507@midway.uchicago.edu> thf2@kimbark.uchicago.edu (Ted Frank) writes:
   In article <1993Apr3.223215.20655@colorado.edu> ajteel@dendrite.cs.Colorado.EDU (A.J. Teel) writes:
   >	Well, the two nifty letters giving concrete proof that the
   >Income Tax is voluntary and giving specific procedures for stopping
   >withholding, et cetera have been out there for a while now.
   >	There has been no refutation to date. Have the nay-sayers
   >finally given up as defeated? Sure would like to hear there reasons
   >for disbelief at this point.

   Probably because you have yet to respond to the refutation I've posted.
   Teel, it's bad enough you post this bs, it's even worse that you don't
   even try to defend it when it gets torn to pieces, but then posting
   that no one's looked at it and gloating when all facts point to the
   contrary point to a severely deluded mind.

What I found interesting about Conklin's letter is the 
6 cases he has won against the IRS.  Now, assuming that
these cases really exist and were one by him (anyone checked?)
they may have nothing to do with his major tax claim.  The IRS fought
one of his deductions.   Defending your deductions seems puny when
you believe that there is no need to file in the first place!

--
				Bernard Silver
				GTE Laboratories
				bsilver@gte.com
				(617) 466-2663



_______
