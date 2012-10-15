#!/usr/bin/env python2

import os
import libxml2
import pprint
import codecs
import json

PROJECT = "/data/csko/youtube/"

def parse(doc):
    ctxt = doc.xpathNewContext()
    ctxt.xpathRegisterNs("tva", "urn:tva:metadata:2011")
    ctxt.xpathRegisterNs("mpeg7", "urn:tva:mpeg7:2008")
    owner_x = ctxt.xpathEval("/tva:TVAMain/tva:MetadataOriginationInformationTable/tva:MetadataOriginationInformation/tva:RightsOwner")
    basic_description_x = ctxt.xpathEval("/tva:TVAMain/tva:ProgramDescription/tva:ProgramInformationTable/tva:ProgramInformation/tva:BasicDescription")[0]
    url_x = ctxt.xpathEval("/tva:TVAMain/tva:ProgramDescription/tva:ProgramLocationTable/tva:OnDemandService/tva:OnDemandProgram/tva:ProgramURL")[0]
    rev_x = ctxt.xpathEval("/tva:TVAMain/tva:ProgramDescription/tva:ProgramReviewTable/tva:Review")
    users_x = ctxt.xpathEval("/tva:TVAMain/tva:ProgramDescription/tva:ProgramReviewTable/ReviewerData/Reviewer")

    def format_review(ctxt, r):
        ctxt.setContextNode(r)

        spam = ctxt.xpathEval("Spam")[0].content
        adult = ctxt.xpathEval("Adult")[0].content

        text = ctxt.xpathEval("tva:FreeTextReview")[0].content
        text = unicode(text, 'utf-8')

        #user = ctxt.xpathEval("tva:Reviewer/Username")[0].content
        user = ctxt.xpathEval("tva:ReviewReference")[0].content
        user = user[len("http://gdata.youtube.com/feeds/api/users/"):]
        date = ctxt.xpathEval("tva:Reviewer/tva:Publication")[0].content
        #reference = reference[0].content if len(reference) > 0 else None
        return dict(spam=spam, adult=adult, text=text, user=user, date=date)


    def format_user(ctxt, u):
        ctxt.setContextNode(u)
        keys = ['Username', 'SubscriberCount', 'School', 'VideosCommented', 'Relationship', 'VideosUploaded',
                'Gender', 'Age', 'VideosRated', 'FriendsAdded', 'Books', 'Movies', 'Job', 'VideosShared', 'Music',
                'Location', 'VideosFavourited', 'Hometown', 'UserSubscriptionsAdded', 'LastWebAccess', 'UserAccountClosed',
                'UserAccountSuspended']
        res = {}
        for key in keys:
            v = ctxt.xpathEval(key)
            res[key] = v[0].content if len(v) > 0 else None
        return res

    owner = owner_x[0].content

    ctxt.setContextNode(basic_description_x)
    title = ctxt.xpathEval("tva:Title")[0].content
    sypnosis = ctxt.xpathEval("tva:Synopsis")[0].content if len(ctxt.xpathEval("tva:Synopsis")) > 0 else None
    keywords = [kw.content for kw in ctxt.xpathEval("tva:Keyword")]
    genre = ctxt.xpathEval("tva:Genre/tva:Name")[0].content
    url = url_x.content

    reviews = [format_review(ctxt, r) for r in rev_x]

    users = [format_user(ctxt, u) for u in users_x]
    assert len(reviews) == len(users)
    for i, review in enumerate(reviews):
        reviews[i]['user'] = users[i]['Username']
    users = dict([(x['Username'], x) for x in users])

    #reviews = [{'spam': r. for r in rev_x]
    #print owner
    #print title
    #print sypnosis
    #print keywords
    #print genre
    #print basic_description_x
    #print url
    #for r in reviews:
        #print r
    video = dict(owner=owner, title=title, sypnosis=sypnosis, keywords=keywords, genre=genre, url=url, reviews=reviews)

    return video, users

def load_data():
    path1 = PROJECT + "db/teachingdata.updated.set01-2012-10-10/"
    path2 = PROJECT + "db/teachingdata.set02/"
    videos = []
    users = {}
    reviews = []
    for path in [path1]:
        for fname in os.listdir(path):
    #        print path + fname
            data = codecs.open(path + fname, encoding='utf-8-sig').read().encode("utf-8")
    #        print data.encode("utf-8")
            doc = libxml2.parseDoc(data)
            video, userlist = parse(doc)
            doc.freeDoc()

            videos.append(video)
            users.update(userlist)
            reviews += video['reviews']
    return videos, users, reviews

def dump_data(videos, users, reviews, path):
    """Dumps only the reviews into JSON format"""
    with open(PROJECT + path, "w") as f:
        json.dump(reviews, f, sort_keys=True, indent=4)

def main():
    print "Loading data."
    videos, users, reviews = load_data()

    print "Dumping data."
    dump_data(videos, users, reviews, "db/comments.json")

if __name__ == "__main__":
    main()
