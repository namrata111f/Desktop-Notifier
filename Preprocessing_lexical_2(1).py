
# coding: utf-8

# In[ ]:


'''
!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
link = 'https://drive.google.com/open?id=1WwdU5ZA1z0Ko8UoS2h0x3A6E-JJSw2JQ'
fluff, id = link.split('=')
import pandas as pd
downloaded = drive.CreateFile({'id':id}) 
downloaded.GetContentFile('final_lexical_features.csv')  
total_dataset_df = pd.read_csv('final_lexical_features.csv')
'''


# In[1]:


import time
start_time = time.time()
import numpy as np
import pandas as pd
import os
from sklearn.utils import shuffle
from urllib import parse
import posixpath
import re
import ipaddress
from collections import Counter
import math
from sklearn.externals import joblib


# In[2]:


def split_url(url):
    if not parse.urlparse(url.strip()).scheme:
        url = 'http://' + url
    protocol, host, path, params, query, fragment = parse.urlparse(url.strip())
    split_dictionary = {
        'url': host + path + params + query + fragment,
        'protocol': protocol,
        'host': host,
        'path': path,
        'params': params,
        'query': query,
        'fragment': fragment }
    return split_dictionary

def url_tokenizer(url_string):
    character_set = ['.','-','_','/','?','=','@','&','!',':','~',',','+','*','#','$','%']
    for character in character_set:
        url_string = url_string.replace(character,' ')
    token = url_string.strip().split()    
    return list(token)

def count(string, character):
    return string.count(character)

def length(string):
    return len(string)

def email_in_text(text):
    return 1 if re.findall(r'[\w\.-]+@[\w\.-]+', text) else 0

def count_tld(text):
    file = ['.aaa', '.aarp', '.abarth', '.abb', '.abbott', '.abbvie', '.abc', '.able', '.abogado', '.abudhabi', '.ac', '.academy', '.accenture', '.accountant', '.accountants', '.aco', '.active', '.actor', '.ad', '.adac', '.ads', '.adult', '.ae', '.aeg', '.aero', '.aetna', '.af', '.afamilycompany', '.afl', '.africa', '.ag', '.agakhan', '.agency', '.ai', '.aig', '.aigo', '.airbus', '.airforce', '.airtel', '.akdn', '.al', '.alfaromeo', '.alibaba', '.alipay', '.allfinanz', '.allstate', '.ally', '.alsace', '.alstom', '.am', '.americanexpress', '.americanfamily', '.amex', '.amfam', '.amica', '.amsterdam', '.analytics', '.android', '.anquan', '.anz', '.ao', '.aol', '.apartments', '.app', '.apple', '.aq', '.aquarelle', '.ar', '.aramco', '.archi', '.army', '.arpa', '.art', '.arte', '.as', '.asda', '.asia', '.associates', '.at', '.athleta', '.attorney', '.au', '.auction', '.audi', '.audible', '.audio', '.auspost', '.author', '.auto', '.autos', '.avianca', '.aw', '.aws', '.ax', '.axa', '.az', '.azure', '.ba', '.baby', '.baidu', '.banamex', '.bananarepublic', '.band', '.bank', '.bar', '.barcelona', '.barclaycard', '.barclays', '.barefoot', '.bargains', '.baseball', '.basketball', '.bauhaus', '.bayern', '.bb', '.bbc', '.bbt', '.bbva', '.bcg', '.bcn', '.bd', '.be', '.beats', '.beauty', '.beer', '.bentley', '.berlin', '.best', '.bestbuy', '.bet', '.bf', '.bg', '.bh', '.bharti', '.bi', '.bible', '.bid', '.bike', '.bing', '.bingo', '.bio', '.biz', '.bj', '.black', '.blackfriday', '.blanco', '.blockbuster', '.blog', '.bloomberg', '.blue', '.bm', '.bms', '.bmw', '.bn', '.bnl', '.bnpparibas', '.bo', '.boats', '.boehringer', '.bofa', '.bom', '.bond', '.boo', '.book', '.booking', '.boots', '.bosch', '.bostik', '.boston', '.bot', '.boutique', '.box', '.br', '.bradesco', '.bridgestone', '.broadway', '.broker', '.brother', '.brussels', '.bs', '.bt', '.budapest', '.bugatti', '.build', '.builders', '.business', '.buy', '.buzz', '.bv', '.bw', '.by', '.bz', '.bzh', '.ca', '.cab', '.cafe', '.cal', '.call', '.calvinklein', '.cam', '.camera', '.camp', '.cancerresearch', '.canon', '.capetown', '.capital', '.capitalone', '.car', '.caravan', '.cards', '.care', '.career', '.careers', '.cars', '.cartier', '.casa', '.case', '.caseih', '.cash', '.casino', '.cat', '.catering', '.catholic', '.cba', '.cbn', '.cbre', '.cbs', '.cc', '.cd', '.ceb', '.center', '.ceo', '.cern', '.cf', '.cfa', '.cfd', '.cg', '.ch', '.chanel', '.channel', '.chase', '.chat', '.cheap', '.chintai', '.chloe', '.christmas', '.chrome', '.chrysler', '.church', '.ci', '.cipriani', '.circle', '.cisco', '.citadel', '.citi', '.citic', '.city', '.cityeats', '.ck', '.cl', '.claims', '.cleaning', '.click', '.clinic', '.clinique', '.clothing', '.cloud', '.club', '.clubmed', '.cm', '.cn', '.co', '.coach', '.codes', '.coffee', '.college', '.cologne', '.com', '.comcast', '.commbank', '.community', '.company', '.compare', '.computer', '.comsec', '.condos', '.construction', '.consulting', '.contact', '.contractors', '.cooking', '.cookingchannel', '.cool', '.coop', '.corsica', '.country', '.coupon', '.coupons', '.courses', '.cr', '.credit', '.creditcard', '.creditunion', '.cricket', '.crown', '.crs', '.cruise', '.cruises', '.csc', '.cu', '.cuisinella', '.cv', '.cw', '.cx', '.cy', '.cymru', '.cyou', '.cz', '.dabur', '.dad', '.dance', '.data', '.date', '.dating', '.datsun', '.day', '.dclk', '.dds', '.de', '.deal', '.dealer', '.deals', '.degree', '.delivery', '.dell', '.deloitte', '.delta', '.democrat', '.dental', '.dentist', '.desi', '.design', '.dev', '.dhl', '.diamonds', '.diet', '.digital', '.direct', '.directory', '.discount', '.discover', '.dish', '.diy', '.dj', '.dk', '.dm', '.dnp', '.do', '.docs', '.doctor', '.dodge', '.dog', '.doha', '.domains', '.dot', '.download', '.drive', '.dtv', '.dubai', '.duck', '.dunlop', '.duns', '.dupont', '.durban', '.dvag', '.dvr', '.dz', '.earth', '.eat', '.ec', '.eco', '.edeka', '.edu', '.education', '.ee', '.eg', '.email', '.emerck', '.energy', '.engineer', '.engineering', '.enterprises', '.epost', '.epson', '.equipment', '.er', '.ericsson', '.erni', '.es', '.esq', '.estate', '.esurance', '.et', '.eu', '.eurovision', '.eus', '.events', '.everbank', '.exchange', '.expert', '.exposed', '.express', '.extraspace', '.fage', '.fail', '.fairwinds', '.faith', '.family', '.fan', '.fans', '.farm', '.farmers', '.fashion', '.fast', '.fedex', '.feedback', '.ferrari', '.ferrero', '.fi', '.fiat', '.fidelity', '.fido', '.film', '.final', '.finance', '.financial', '.fire', '.firestone', '.firmdale', '.fish', '.fishing', '.fit', '.fitness', '.fj', '.fk', '.flickr', '.flights', '.flir', '.florist', '.flowers', '.fly', '.fm', '.fo', '.foo', '.food', '.foodnetwork', '.football', '.ford', '.forex', '.forsale', '.forum', '.foundation', '.fox', '.fr', '.free', '.fresenius', '.frl', '.frogans', '.frontdoor', '.frontier', '.ftr', '.fujitsu', '.fujixerox', '.fun', '.fund', '.furniture', '.futbol', '.fyi', '.ga', '.gal', '.gallery', '.gallo', '.gallup', '.game', '.games', '.gap', '.garden', '.gb', '.gbiz', '.gd', '.gdn', '.ge', '.gea', '.gent', '.genting', '.george', '.gf', '.gg', '.ggee', '.gh', '.gi', '.gift', '.gifts', '.gives', '.giving', '.gl', '.glade', '.glass', '.gle', '.global', '.globo', '.gm', '.gmail', '.gmbh', '.gmo', '.gmx', '.gn', '.godaddy', '.gold', '.goldpoint', '.golf', '.goo', '.goodhands', '.goodyear', '.goog', '.google', '.gop', '.got', '.gov', '.gp', '.gq', '.gr', '.grainger', '.graphics', '.gratis', '.green', '.gripe', '.group', '.gs', '.gt', '.gu', '.guardian', '.gucci', '.guge', '.guide', '.guitars', '.guru', '.gw', '.gy', '.hair', '.hamburg', '.hangout', '.haus', '.hbo', '.hdfc', '.hdfcbank', '.health', '.healthcare', '.help', '.helsinki', '.here', '.hermes', '.hgtv', '.hiphop', '.hisamitsu', '.hitachi', '.hiv', '.hk', '.hkt', '.hm', '.hn', '.hockey', '.holdings', '.holiday', '.homedepot', '.homegoods', '.homes', '.homesense', '.honda', '.honeywell', '.horse', '.hospital', '.host', '.hosting', '.hot', '.hoteles', '.hotmail', '.house', '.how', '.hr', '.hsbc', '.ht', '.htc', '.hu', '.hughes', '.hyatt', '.hyundai', '.ibm', '.icbc', '.ice', '.icu', '.id', '.ie', '.ieee', '.ifm', '.ikano', '.il', '.im', '.imamat', '.imdb', '.immo', '.immobilien', '.in', '.industries', '.infiniti', '.info', '.ing', '.ink', '.institute', '.insurance', '.insure', '.int', '.intel', '.international', '.intuit', '.investments', '.io', '.ipiranga', '.iq', '.ir', '.irish', '.is', '.iselect', '.ismaili', '.ist', '.istanbul', '.it', '.itau', '.itv', '.iveco', '.iwc', '.jaguar', '.java', '.jcb', '.jcp', '.je', '.jeep', '.jetzt', '.jewelry', '.jio', '.jlc', '.jll', '.jm', '.jmp', '.jnj', '.jo', '.jobs', '.joburg', '.jot', '.joy', '.jp', '.jpmorgan', '.jprs', '.juegos', '.juniper', '.kaufen', '.kddi', '.ke', '.kerryhotels', '.kerrylogistics', '.kerryproperties', '.kfh', '.kg', '.kh', '.ki', '.kia', '.kim', '.kinder', '.kindle', '.kitchen', '.kiwi', '.km', '.kn', '.koeln', '.komatsu', '.kosher', '.kp', '.kpmg', '.kpn', '.kr', '.krd', '.kred', '.kuokgroup', '.kw', '.ky', '.kyoto', '.kz', '.la', '.lacaixa', '.ladbrokes', '.lamborghini', '.lamer', '.lancaster', '.lancia', '.lancome', '.land', '.landrover', '.lanxess', '.lasalle', '.lat', '.latino', '.latrobe', '.law', '.lawyer', '.lb', '.lc', '.lds', '.lease', '.leclerc', '.lefrak', '.legal', '.lego', '.lexus', '.lgbt', '.li', '.liaison', '.lidl', '.life', '.lifeinsurance', '.lifestyle', '.lighting', '.like', '.lilly', '.limited', '.limo', '.lincoln', '.linde', '.link', '.lipsy', '.live', '.living', '.lixil', '.lk', '.loan', '.loans', '.locker', '.locus', '.loft', '.lol', '.london', '.lotte', '.lotto', '.love', '.lpl', '.lplfinancial', '.lr', '.ls', '.lt', '.ltd', '.ltda', '.lu', '.lundbeck', '.lupin', '.luxe', '.luxury', '.lv', '.ly', '.ma', '.macys', '.madrid', '.maif', '.maison', '.makeup', '.man', '.management', '.mango', '.market', '.marketing', '.markets', '.marriott', '.marshalls', '.maserati', '.mattel', '.mba', '.mc', '.mcd', '.mcdonalds', '.mckinsey', '.md', '.me', '.med', '.media', '.meet', '.melbourne', '.meme', '.memorial', '.men', '.menu', '.meo', '.metlife', '.mg', '.mh', '.miami', '.microsoft', '.mil', '.mini', '.mint', '.mit', '.mitsubishi', '.mk', '.ml', '.mlb', '.mls', '.mm', '.mma', '.mn', '.mo', '.mobi', '.mobile', '.mobily', '.moda', '.moe', '.moi', '.mom', '.monash', '.money', '.monster', '.montblanc', '.mopar', '.mormon', '.mortgage', '.moscow', '.moto', '.motorcycles', '.mov', '.movie', '.movistar', '.mp', '.mq', '.mr', '.ms', '.msd', '.mt', '.mtn', '.mtpc', '.mtr', '.mu', '.museum', '.mutual', '.mv', '.mw', '.mx', '.my', '.mz', '.na', '.nab', '.nadex', '.nagoya', '.name', '.nationwide', '.natura', '.navy', '.nba', '.nc', '.ne', '.nec', '.net', '.netbank', '.netflix', '.network', '.neustar', '.new', '.newholland', '.news', '.next', '.nextdirect', '.nexus', '.nf', '.nfl', '.ng', '.ngo', '.nhk', '.ni', '.nico', '.nike', '.nikon', '.ninja', '.nissan', '.nissay', '.nl', '.no', '.nokia', '.northwesternmutual', '.norton', '.now', '.nowruz', '.nowtv', '.np', '.nr', '.nra', '.nrw', '.ntt', '.nu', '.nyc', '.nz', '.obi', '.observer', '.off', '.office', '.okinawa', '.olayan', '.olayangroup', '.oldnavy', '.ollo', '.om', '.omega', '.one', '.ong', '.onl', '.online', '.onyourside', '.ooo', '.open', '.oracle', '.orange', '.org', '.organic', '.orientexpress', '.origins', '.osaka', '.otsuka', '.ott', '.ovh', '.pa', '.page', '.pamperedchef', '.panasonic', '.panerai', '.paris', '.pars', '.partners', '.parts', '.party', '.passagens', '.pay', '.pccw', '.pe', '.pet', '.pf', '.pfizer', '.pg', '.ph', '.pharmacy', '.philips', '.phone', '.photo', '.photography', '.photos', '.physio', '.piaget', '.pics', '.pictet', '.pictures', '.pid', '.pin', '.ping', '.pink', '.pioneer', '.pizza', '.pk', '.pl', '.place', '.play', '.playstation', '.plumbing', '.plus', '.pm', '.pn', '.pnc', '.pohl', '.poker', '.politie', '.porn', '.post', '.pr', '.pramerica', '.praxi', '.press', '.prime', '.pro', '.prod', '.productions', '.prof', '.progressive', '.promo', '.properties', '.property', '.protection', '.pru', '.prudential', '.ps', '.pt', '.pub', '.pw', '.pwc', '.py', '.qa', '.qpon', '.quebec', '.quest', '.qvc', '.racing', '.radio', '.raid', '.re', '.read', '.realestate', '.realtor', '.realty', '.recipes', '.red', '.redstone', '.redumbrella', '.rehab', '.reise', '.reisen', '.reit', '.reliance', '.ren', '.rent', '.rentals', '.repair', '.report', '.republican', '.rest', '.restaurant', '.review', '.reviews', '.rexroth', '.rich', '.richardli', '.ricoh', '.rightathome', '.ril', '.rio', '.rip', '.rmit', '.ro', '.rocher', '.rocks', '.rodeo', '.rogers', '.room', '.rs', '.rsvp', '.ru', '.ruhr', '.run', '.rw', '.rwe', '.ryukyu', '.sa', '.saarland', '.safe', '.safety', '.sakura', '.sale', '.salon', '.samsclub', '.samsung', '.sandvik', '.sandvikcoromant', '.sanofi', '.sap', '.sapo', '.sarl', '.sas', '.save', '.saxo', '.sb', '.sbi', '.sbs', '.sc', '.sca', '.scb', '.schaeffler', '.schmidt', '.scholarships', '.school', '.schule', '.schwarz', '.science', '.scjohnson', '.scor', '.scot', '.sd', '.se', '.seat', '.secure', '.security', '.seek', '.select', '.sener', '.services', '.ses', '.seven', '.sew', '.sex', '.sexy', '.sfr', '.sg', '.sh', '.shangrila', '.sharp', '.shaw', '.shell', '.shia', '.shiksha', '.shoes', '.shop', '.shopping', '.shouji', '.show', '.showtime', '.shriram', '.si', '.silk', '.sina', '.singles', '.site', '.sj', '.sk', '.ski', '.skin', '.sky', '.skype', '.sl', '.sling', '.sm', '.smart', '.smile', '.sn', '.sncf', '.so', '.soccer', '.social', '.softbank', '.software', '.sohu', '.solar', '.solutions', '.song', '.sony', '.soy', '.space', '.spiegel', '.spot', '.spreadbetting', '.sr', '.srl', '.srt', '.st', '.stada', '.staples', '.star', '.starhub', '.statebank', '.statefarm', '.statoil', '.stc', '.stcgroup', '.stockholm', '.storage', '.store', '.stream', '.studio', '.study', '.style', '.su', '.sucks', '.supplies', '.supply', '.support', '.surf', '.surgery', '.suzuki', '.sv', '.swatch', '.swiftcover', '.swiss', '.sx', '.sy', '.sydney', '.symantec', '.systems', '.sz', '.tab', '.taipei', '.talk', '.taobao', '.target', '.tatamotors', '.tatar', '.tattoo', '.tax', '.taxi', '.tc', '.tci', '.td', '.tdk', '.team', '.tech', '.technology', '.tel', '.telecity', '.telefonica', '.temasek', '.tennis', '.teva', '.tf', '.tg', '.th', '.thd', '.theater', '.theatre', '.tiaa', '.tickets', '.tienda', '.tiffany', '.tips', '.tires', '.tirol', '.tj', '.tjmaxx', '.tjx', '.tk', '.tkmaxx', '.tl', '.tm', '.tmall', '.tn', '.to', '.today', '.tokyo', '.tools', '.top', '.toray', '.toshiba', '.total', '.tours', '.town', '.toyota', '.toys', '.tr', '.trade', '.trading', '.training', '.travel', '.travelchannel', '.travelers', '.travelersinsurance', '.trust', '.trv', '.tt', '.tube', '.tui', '.tunes', '.tushu', '.tv', '.tvs', '.tw', '.tz', '.ua', '.ubank', '.ubs', '.uconnect', '.ug', '.uk', '.unicom', '.university', '.uno', '.uol', '.ups', '.us', '.uy', '.uz', '.va', '.vacations', '.vana', '.vanguard', '.vc', '.ve', '.vegas', '.ventures', '.verisign', '.versicherung', '.vet', '.vg', '.vi', '.viajes', '.video', '.vig', '.viking', '.villas', '.vin', '.vip', '.virgin', '.visa', '.vision', '.vista', '.vistaprint', '.viva', '.vivo', '.vlaanderen', '.vn', '.vodka', '.volkswagen', '.volvo', '.vote', '.voting', '.voto', '.voyage', '.vu', '.vuelos', '.wales', '.walmart', '.walter', '.wang', '.wanggou', '.warman', '.watch', '.watches', '.weather', '.weatherchannel', '.webcam', '.weber', '.website', '.wed', '.wedding', '.weibo', '.weir', '.wf', '.whoswho', '.wien', '.wiki', '.williamhill', '.win', '.windows', '.wine', '.winners', '.wme', '.wolterskluwer', '.woodside', '.work', '.works', '.world', '.wow', '.ws', '.wtc', '.wtf', '.xbox', '.xerox', '.xfinity', '.xihuan', '.xin', '.xperia', '.xxx', '.xyz', '.achts', '.yahoo', '.yamaxun', '.yandex', '.ye', '.yodobashi', '.yoga', '.yokohama', '.you', '.youtube', '.yt', '.yun', '.za', '.zappos', '.zara', '.zero', '.zip', '.zippo', '.zm', '.zone', '.zuerich', '.zw']
    c = 0
    pattern = re.compile("[a-zA-Z0-9.]")
    for line in file:
        L = len(line)
        i = (text.lower().strip()).find(line.strip())
        while i > -1:
            if ((i+L-1) >= len(text)) or not pattern.match(text[i+L-1]):
                c = c + 1
            i = text.find(line.strip(), i + 1)
    return c

def check_tld(text):
    file = ['.aaa', '.aarp', '.abarth', '.abb', '.abbott', '.abbvie', '.abc', '.able', '.abogado', '.abudhabi', '.ac', '.academy', '.accenture', '.accountant', '.accountants', '.aco', '.active', '.actor', '.ad', '.adac', '.ads', '.adult', '.ae', '.aeg', '.aero', '.aetna', '.af', '.afamilycompany', '.afl', '.africa', '.ag', '.agakhan', '.agency', '.ai', '.aig', '.aigo', '.airbus', '.airforce', '.airtel', '.akdn', '.al', '.alfaromeo', '.alibaba', '.alipay', '.allfinanz', '.allstate', '.ally', '.alsace', '.alstom', '.am', '.americanexpress', '.americanfamily', '.amex', '.amfam', '.amica', '.amsterdam', '.analytics', '.android', '.anquan', '.anz', '.ao', '.aol', '.apartments', '.app', '.apple', '.aq', '.aquarelle', '.ar', '.aramco', '.archi', '.army', '.arpa', '.art', '.arte', '.as', '.asda', '.asia', '.associates', '.at', '.athleta', '.attorney', '.au', '.auction', '.audi', '.audible', '.audio', '.auspost', '.author', '.auto', '.autos', '.avianca', '.aw', '.aws', '.ax', '.axa', '.az', '.azure', '.ba', '.baby', '.baidu', '.banamex', '.bananarepublic', '.band', '.bank', '.bar', '.barcelona', '.barclaycard', '.barclays', '.barefoot', '.bargains', '.baseball', '.basketball', '.bauhaus', '.bayern', '.bb', '.bbc', '.bbt', '.bbva', '.bcg', '.bcn', '.bd', '.be', '.beats', '.beauty', '.beer', '.bentley', '.berlin', '.best', '.bestbuy', '.bet', '.bf', '.bg', '.bh', '.bharti', '.bi', '.bible', '.bid', '.bike', '.bing', '.bingo', '.bio', '.biz', '.bj', '.black', '.blackfriday', '.blanco', '.blockbuster', '.blog', '.bloomberg', '.blue', '.bm', '.bms', '.bmw', '.bn', '.bnl', '.bnpparibas', '.bo', '.boats', '.boehringer', '.bofa', '.bom', '.bond', '.boo', '.book', '.booking', '.boots', '.bosch', '.bostik', '.boston', '.bot', '.boutique', '.box', '.br', '.bradesco', '.bridgestone', '.broadway', '.broker', '.brother', '.brussels', '.bs', '.bt', '.budapest', '.bugatti', '.build', '.builders', '.business', '.buy', '.buzz', '.bv', '.bw', '.by', '.bz', '.bzh', '.ca', '.cab', '.cafe', '.cal', '.call', '.calvinklein', '.cam', '.camera', '.camp', '.cancerresearch', '.canon', '.capetown', '.capital', '.capitalone', '.car', '.caravan', '.cards', '.care', '.career', '.careers', '.cars', '.cartier', '.casa', '.case', '.caseih', '.cash', '.casino', '.cat', '.catering', '.catholic', '.cba', '.cbn', '.cbre', '.cbs', '.cc', '.cd', '.ceb', '.center', '.ceo', '.cern', '.cf', '.cfa', '.cfd', '.cg', '.ch', '.chanel', '.channel', '.chase', '.chat', '.cheap', '.chintai', '.chloe', '.christmas', '.chrome', '.chrysler', '.church', '.ci', '.cipriani', '.circle', '.cisco', '.citadel', '.citi', '.citic', '.city', '.cityeats', '.ck', '.cl', '.claims', '.cleaning', '.click', '.clinic', '.clinique', '.clothing', '.cloud', '.club', '.clubmed', '.cm', '.cn', '.co', '.coach', '.codes', '.coffee', '.college', '.cologne', '.com', '.comcast', '.commbank', '.community', '.company', '.compare', '.computer', '.comsec', '.condos', '.construction', '.consulting', '.contact', '.contractors', '.cooking', '.cookingchannel', '.cool', '.coop', '.corsica', '.country', '.coupon', '.coupons', '.courses', '.cr', '.credit', '.creditcard', '.creditunion', '.cricket', '.crown', '.crs', '.cruise', '.cruises', '.csc', '.cu', '.cuisinella', '.cv', '.cw', '.cx', '.cy', '.cymru', '.cyou', '.cz', '.dabur', '.dad', '.dance', '.data', '.date', '.dating', '.datsun', '.day', '.dclk', '.dds', '.de', '.deal', '.dealer', '.deals', '.degree', '.delivery', '.dell', '.deloitte', '.delta', '.democrat', '.dental', '.dentist', '.desi', '.design', '.dev', '.dhl', '.diamonds', '.diet', '.digital', '.direct', '.directory', '.discount', '.discover', '.dish', '.diy', '.dj', '.dk', '.dm', '.dnp', '.do', '.docs', '.doctor', '.dodge', '.dog', '.doha', '.domains', '.dot', '.download', '.drive', '.dtv', '.dubai', '.duck', '.dunlop', '.duns', '.dupont', '.durban', '.dvag', '.dvr', '.dz', '.earth', '.eat', '.ec', '.eco', '.edeka', '.edu', '.education', '.ee', '.eg', '.email', '.emerck', '.energy', '.engineer', '.engineering', '.enterprises', '.epost', '.epson', '.equipment', '.er', '.ericsson', '.erni', '.es', '.esq', '.estate', '.esurance', '.et', '.eu', '.eurovision', '.eus', '.events', '.everbank', '.exchange', '.expert', '.exposed', '.express', '.extraspace', '.fage', '.fail', '.fairwinds', '.faith', '.family', '.fan', '.fans', '.farm', '.farmers', '.fashion', '.fast', '.fedex', '.feedback', '.ferrari', '.ferrero', '.fi', '.fiat', '.fidelity', '.fido', '.film', '.final', '.finance', '.financial', '.fire', '.firestone', '.firmdale', '.fish', '.fishing', '.fit', '.fitness', '.fj', '.fk', '.flickr', '.flights', '.flir', '.florist', '.flowers', '.fly', '.fm', '.fo', '.foo', '.food', '.foodnetwork', '.football', '.ford', '.forex', '.forsale', '.forum', '.foundation', '.fox', '.fr', '.free', '.fresenius', '.frl', '.frogans', '.frontdoor', '.frontier', '.ftr', '.fujitsu', '.fujixerox', '.fun', '.fund', '.furniture', '.futbol', '.fyi', '.ga', '.gal', '.gallery', '.gallo', '.gallup', '.game', '.games', '.gap', '.garden', '.gb', '.gbiz', '.gd', '.gdn', '.ge', '.gea', '.gent', '.genting', '.george', '.gf', '.gg', '.ggee', '.gh', '.gi', '.gift', '.gifts', '.gives', '.giving', '.gl', '.glade', '.glass', '.gle', '.global', '.globo', '.gm', '.gmail', '.gmbh', '.gmo', '.gmx', '.gn', '.godaddy', '.gold', '.goldpoint', '.golf', '.goo', '.goodhands', '.goodyear', '.goog', '.google', '.gop', '.got', '.gov', '.gp', '.gq', '.gr', '.grainger', '.graphics', '.gratis', '.green', '.gripe', '.group', '.gs', '.gt', '.gu', '.guardian', '.gucci', '.guge', '.guide', '.guitars', '.guru', '.gw', '.gy', '.hair', '.hamburg', '.hangout', '.haus', '.hbo', '.hdfc', '.hdfcbank', '.health', '.healthcare', '.help', '.helsinki', '.here', '.hermes', '.hgtv', '.hiphop', '.hisamitsu', '.hitachi', '.hiv', '.hk', '.hkt', '.hm', '.hn', '.hockey', '.holdings', '.holiday', '.homedepot', '.homegoods', '.homes', '.homesense', '.honda', '.honeywell', '.horse', '.hospital', '.host', '.hosting', '.hot', '.hoteles', '.hotmail', '.house', '.how', '.hr', '.hsbc', '.ht', '.htc', '.hu', '.hughes', '.hyatt', '.hyundai', '.ibm', '.icbc', '.ice', '.icu', '.id', '.ie', '.ieee', '.ifm', '.ikano', '.il', '.im', '.imamat', '.imdb', '.immo', '.immobilien', '.in', '.industries', '.infiniti', '.info', '.ing', '.ink', '.institute', '.insurance', '.insure', '.int', '.intel', '.international', '.intuit', '.investments', '.io', '.ipiranga', '.iq', '.ir', '.irish', '.is', '.iselect', '.ismaili', '.ist', '.istanbul', '.it', '.itau', '.itv', '.iveco', '.iwc', '.jaguar', '.java', '.jcb', '.jcp', '.je', '.jeep', '.jetzt', '.jewelry', '.jio', '.jlc', '.jll', '.jm', '.jmp', '.jnj', '.jo', '.jobs', '.joburg', '.jot', '.joy', '.jp', '.jpmorgan', '.jprs', '.juegos', '.juniper', '.kaufen', '.kddi', '.ke', '.kerryhotels', '.kerrylogistics', '.kerryproperties', '.kfh', '.kg', '.kh', '.ki', '.kia', '.kim', '.kinder', '.kindle', '.kitchen', '.kiwi', '.km', '.kn', '.koeln', '.komatsu', '.kosher', '.kp', '.kpmg', '.kpn', '.kr', '.krd', '.kred', '.kuokgroup', '.kw', '.ky', '.kyoto', '.kz', '.la', '.lacaixa', '.ladbrokes', '.lamborghini', '.lamer', '.lancaster', '.lancia', '.lancome', '.land', '.landrover', '.lanxess', '.lasalle', '.lat', '.latino', '.latrobe', '.law', '.lawyer', '.lb', '.lc', '.lds', '.lease', '.leclerc', '.lefrak', '.legal', '.lego', '.lexus', '.lgbt', '.li', '.liaison', '.lidl', '.life', '.lifeinsurance', '.lifestyle', '.lighting', '.like', '.lilly', '.limited', '.limo', '.lincoln', '.linde', '.link', '.lipsy', '.live', '.living', '.lixil', '.lk', '.loan', '.loans', '.locker', '.locus', '.loft', '.lol', '.london', '.lotte', '.lotto', '.love', '.lpl', '.lplfinancial', '.lr', '.ls', '.lt', '.ltd', '.ltda', '.lu', '.lundbeck', '.lupin', '.luxe', '.luxury', '.lv', '.ly', '.ma', '.macys', '.madrid', '.maif', '.maison', '.makeup', '.man', '.management', '.mango', '.market', '.marketing', '.markets', '.marriott', '.marshalls', '.maserati', '.mattel', '.mba', '.mc', '.mcd', '.mcdonalds', '.mckinsey', '.md', '.me', '.med', '.media', '.meet', '.melbourne', '.meme', '.memorial', '.men', '.menu', '.meo', '.metlife', '.mg', '.mh', '.miami', '.microsoft', '.mil', '.mini', '.mint', '.mit', '.mitsubishi', '.mk', '.ml', '.mlb', '.mls', '.mm', '.mma', '.mn', '.mo', '.mobi', '.mobile', '.mobily', '.moda', '.moe', '.moi', '.mom', '.monash', '.money', '.monster', '.montblanc', '.mopar', '.mormon', '.mortgage', '.moscow', '.moto', '.motorcycles', '.mov', '.movie', '.movistar', '.mp', '.mq', '.mr', '.ms', '.msd', '.mt', '.mtn', '.mtpc', '.mtr', '.mu', '.museum', '.mutual', '.mv', '.mw', '.mx', '.my', '.mz', '.na', '.nab', '.nadex', '.nagoya', '.name', '.nationwide', '.natura', '.navy', '.nba', '.nc', '.ne', '.nec', '.net', '.netbank', '.netflix', '.network', '.neustar', '.new', '.newholland', '.news', '.next', '.nextdirect', '.nexus', '.nf', '.nfl', '.ng', '.ngo', '.nhk', '.ni', '.nico', '.nike', '.nikon', '.ninja', '.nissan', '.nissay', '.nl', '.no', '.nokia', '.northwesternmutual', '.norton', '.now', '.nowruz', '.nowtv', '.np', '.nr', '.nra', '.nrw', '.ntt', '.nu', '.nyc', '.nz', '.obi', '.observer', '.off', '.office', '.okinawa', '.olayan', '.olayangroup', '.oldnavy', '.ollo', '.om', '.omega', '.one', '.ong', '.onl', '.online', '.onyourside', '.ooo', '.open', '.oracle', '.orange', '.org', '.organic', '.orientexpress', '.origins', '.osaka', '.otsuka', '.ott', '.ovh', '.pa', '.page', '.pamperedchef', '.panasonic', '.panerai', '.paris', '.pars', '.partners', '.parts', '.party', '.passagens', '.pay', '.pccw', '.pe', '.pet', '.pf', '.pfizer', '.pg', '.ph', '.pharmacy', '.philips', '.phone', '.photo', '.photography', '.photos', '.physio', '.piaget', '.pics', '.pictet', '.pictures', '.pid', '.pin', '.ping', '.pink', '.pioneer', '.pizza', '.pk', '.pl', '.place', '.play', '.playstation', '.plumbing', '.plus', '.pm', '.pn', '.pnc', '.pohl', '.poker', '.politie', '.porn', '.post', '.pr', '.pramerica', '.praxi', '.press', '.prime', '.pro', '.prod', '.productions', '.prof', '.progressive', '.promo', '.properties', '.property', '.protection', '.pru', '.prudential', '.ps', '.pt', '.pub', '.pw', '.pwc', '.py', '.qa', '.qpon', '.quebec', '.quest', '.qvc', '.racing', '.radio', '.raid', '.re', '.read', '.realestate', '.realtor', '.realty', '.recipes', '.red', '.redstone', '.redumbrella', '.rehab', '.reise', '.reisen', '.reit', '.reliance', '.ren', '.rent', '.rentals', '.repair', '.report', '.republican', '.rest', '.restaurant', '.review', '.reviews', '.rexroth', '.rich', '.richardli', '.ricoh', '.rightathome', '.ril', '.rio', '.rip', '.rmit', '.ro', '.rocher', '.rocks', '.rodeo', '.rogers', '.room', '.rs', '.rsvp', '.ru', '.ruhr', '.run', '.rw', '.rwe', '.ryukyu', '.sa', '.saarland', '.safe', '.safety', '.sakura', '.sale', '.salon', '.samsclub', '.samsung', '.sandvik', '.sandvikcoromant', '.sanofi', '.sap', '.sapo', '.sarl', '.sas', '.save', '.saxo', '.sb', '.sbi', '.sbs', '.sc', '.sca', '.scb', '.schaeffler', '.schmidt', '.scholarships', '.school', '.schule', '.schwarz', '.science', '.scjohnson', '.scor', '.scot', '.sd', '.se', '.seat', '.secure', '.security', '.seek', '.select', '.sener', '.services', '.ses', '.seven', '.sew', '.sex', '.sexy', '.sfr', '.sg', '.sh', '.shangrila', '.sharp', '.shaw', '.shell', '.shia', '.shiksha', '.shoes', '.shop', '.shopping', '.shouji', '.show', '.showtime', '.shriram', '.si', '.silk', '.sina', '.singles', '.site', '.sj', '.sk', '.ski', '.skin', '.sky', '.skype', '.sl', '.sling', '.sm', '.smart', '.smile', '.sn', '.sncf', '.so', '.soccer', '.social', '.softbank', '.software', '.sohu', '.solar', '.solutions', '.song', '.sony', '.soy', '.space', '.spiegel', '.spot', '.spreadbetting', '.sr', '.srl', '.srt', '.st', '.stada', '.staples', '.star', '.starhub', '.statebank', '.statefarm', '.statoil', '.stc', '.stcgroup', '.stockholm', '.storage', '.store', '.stream', '.studio', '.study', '.style', '.su', '.sucks', '.supplies', '.supply', '.support', '.surf', '.surgery', '.suzuki', '.sv', '.swatch', '.swiftcover', '.swiss', '.sx', '.sy', '.sydney', '.symantec', '.systems', '.sz', '.tab', '.taipei', '.talk', '.taobao', '.target', '.tatamotors', '.tatar', '.tattoo', '.tax', '.taxi', '.tc', '.tci', '.td', '.tdk', '.team', '.tech', '.technology', '.tel', '.telecity', '.telefonica', '.temasek', '.tennis', '.teva', '.tf', '.tg', '.th', '.thd', '.theater', '.theatre', '.tiaa', '.tickets', '.tienda', '.tiffany', '.tips', '.tires', '.tirol', '.tj', '.tjmaxx', '.tjx', '.tk', '.tkmaxx', '.tl', '.tm', '.tmall', '.tn', '.to', '.today', '.tokyo', '.tools', '.top', '.toray', '.toshiba', '.total', '.tours', '.town', '.toyota', '.toys', '.tr', '.trade', '.trading', '.training', '.travel', '.travelchannel', '.travelers', '.travelersinsurance', '.trust', '.trv', '.tt', '.tube', '.tui', '.tunes', '.tushu', '.tv', '.tvs', '.tw', '.tz', '.ua', '.ubank', '.ubs', '.uconnect', '.ug', '.uk', '.unicom', '.university', '.uno', '.uol', '.ups', '.us', '.uy', '.uz', '.va', '.vacations', '.vana', '.vanguard', '.vc', '.ve', '.vegas', '.ventures', '.verisign', '.versicherung', '.vet', '.vg', '.vi', '.viajes', '.video', '.vig', '.viking', '.villas', '.vin', '.vip', '.virgin', '.visa', '.vision', '.vista', '.vistaprint', '.viva', '.vivo', '.vlaanderen', '.vn', '.vodka', '.volkswagen', '.volvo', '.vote', '.voting', '.voto', '.voyage', '.vu', '.vuelos', '.wales', '.walmart', '.walter', '.wang', '.wanggou', '.warman', '.watch', '.watches', '.weather', '.weatherchannel', '.webcam', '.weber', '.website', '.wed', '.wedding', '.weibo', '.weir', '.wf', '.whoswho', '.wien', '.wiki', '.williamhill', '.win', '.windows', '.wine', '.winners', '.wme', '.wolterskluwer', '.woodside', '.work', '.works', '.world', '.wow', '.ws', '.wtc', '.wtf', '.xbox', '.xerox', '.xfinity', '.xihuan', '.xin', '.xperia', '.xxx', '.xyz', '.achts', '.yahoo', '.yamaxun', '.yandex', '.ye', '.yodobashi', '.yoga', '.yokohama', '.you', '.youtube', '.yt', '.yun', '.za', '.zappos', '.zara', '.zero', '.zip', '.zippo', '.zm', '.zone', '.zuerich', '.zw']
    pattern = re.compile("[a-zA-Z0-9.]")
    for line in file:
        i = (text.lower().strip()).find(line.strip())
        while i > -1:
            if ((i + len(line) - 1) >= len(text)) or not pattern.match(text[i + len(line) - 1]):
                return 1
            i = text.find(line.strip(), i + 1)
    return 0

def extract_extension(text):
    file = ['apk', 'asp', 'aspx', 'avi', 'bat', 'bin', 'c', 'cab', 'cfm', 'cgi', 'class', 'com', 'cpl', 'cpp', 'css', 'dat', 'dds', 'dll', 'doc', 'docx', 'exe', 'gif', 'gz', 'h', 'htm', 'html', 'ico', 'jar', 'jpg', 'js', 'jsp', 'lua', 'm', 'mov', 'mp3', 'mp4', 'mpg', 'pdf', 'php', 'pl', 'png', 'py', 'rar', 'rss', 'sh', 'svg', 'swf', 'sys', 'tar', 'tmp', 'torrent', 'txt', 'xhtml', 'xls', 'xml', 'zip']
    pattern = re.compile("[a-zA-Z0-9.]")
    for extension in file:
        i = (text.lower().strip()).find(extension.strip())
        while i > -1:
            if ((i + len(extension) - 1) >= len(text)) or not pattern.match(text[i + len(extension) - 1]):
                return extension.rstrip().split('.')[-1]
            i = text.find(extension.strip(), i + 1)
    return '?'

def count_params(text):
    return len(parse.parse_qs(text))

def check_server_client(text):
    if "server" in text.lower() or "client" in text.lower():
        return 1
    return 0

def count_vowels(string):
    return sum(string.lower().count(c) for c in ['a','e','i','o','u'])

def check_url_shortener(url_token):
    file = ['0rz.tw', '1-url.net', '126.am', '1tk.us', '1un.fr', '1url.com', '1url.cz', '1wb2.net', '2.gp', '2.ht', '2ad.in', '2doc.net', '2fear.com', '2tu.us', '2ty.in', '2u.xf.cz', '3ra.be', '3x.si', '4i.ae', '4ks.net', '4view.me', '5em.cz', '5url.net', '5z8.info', '6fr.ru', '6g6.eu', '7.ly', '76.gd', '77.ai', '7fth.cc', '7li.in', '7vd.cn', '8u.cz', '944.la', '98.to', 'L9.fr', 'Lvvk.com', 'To8.cc', 'a0.fr', 'abbr.sk', 'ad-med.cz', 'ad5.eu', 'ad7.biz', 'adb.ug', 'adf.ly', 'adfa.st', 'adfly.fr', 'adli.pw', 'adv.li', 'ajn.me', 'aka.gr', 'alil.in', 'amzn.to', 'any.gs', 'aqva.pl', 'ares.tl', 'asso.in', 'au.ms', 'ayt.fr', 'azali.fr', 'b00.fr', 'b23.ru', 'b54.in', 'baid.us', 'bc.vc', 'beam.to', 'bee4.biz', 'bim.im', 'bit.do', 'bit.ly', 'bitly.com', 'bitw.in', 'blap.net', 'ble.pl', 'blip.tv', 'boi.re', 'bote.me', 'bougn.at', 'br4.in', 'brk.to', 'brzu.net', 'bul.lu', 'bxl.me', 'bzh.me', 'cachor.ro', 'captur.in', 'cashfly.com', 'cbs.so', 'cbug.cc', 'cc.cc', 'ccj.im', 'cf.ly', 'cf2.me', 'cf6.co', 'chilp.it', 'cjb.net', 'cli.gs', 'clikk.in', 'clk.im', 'cn86.org', 'couic.fr', 'cr.tl', 'cudder.it', 'cur.lv', 'curl.im', 'curte.me', 'cut.pe', 'cut.sk', 'cutt.eu', 'cutt.us', 'cutu.me', 'cybr.fr', 'cyonix.to', 'd75.eu', 'daa.pl', 'dai.ly', 'decenturl.com', 'dd.ma', 'ddp.net', 'dft.ba', 'digbig.com', 'doiop.com', 'dolp.cc', 'dopice.sk', 'droid.ws', 'dv.gd', 'dyo.gs', 'e37.eu', 'easyurl.net', 'ecra.se', 'ely.re', 'encurtador.com.br', 'erax.cz', 'erw.cz', 'esy.es', 'ex9.co', 'ezurl.cc', 'fff.re', 'fff.to', 'fff.wf', 'filz.fr', 'fnk.es', 'foe.hn', 'folu.me', 'freze.it', 'fur.ly', 'fwdurl.net', 'g00.me', 'gca.sh', 'gg.gg', 'goo.gl', 'goo.lu', 'grem.io', 'guiama.is', 'hadej.co', 'hide.my', 'hjkl.fr', 'hops.me', 'href.li', 'ht.ly', 'i-2.co', 'i99.cz', 'icit.fr', 'ick.li', 'icks.ro', 'iiiii.in', 'iky.fr', 'ilix.in', 'info.ms', 'is.gd', 'isra.li', 'itm.im', 'ity.im', 'ix.sk', 'j.gs', 'j.mp', 'jdem.cz', 'jieb.be', 'jp22.net', 'jqw.de', 'kask.us', 'kd2.org', 'kfd.pl', 'korta.nu', 'kr3w.de', 'krat.si', 'kratsi.cz', 'krod.cz', 'kuc.cz', 'kxb.me', 'l-k.be', 'lc-s.co', 'lc.cx', 'lcut.in', 'letop10.', 'libero.it', 'lick.my', 'lien.li', 'lien.pl', 'lin.io', 'linkn.co', 'linkbucks.com', 'llu.ch', 'lnk.co', 'lnk.ly', 'lnk.sk', 'lnks.fr', 'lnky.fr', 'lnp.sn', 'lp25.fr', 'm1p.fr', 'm3mi.com', 'make.my', 'mcaf.ee', 'mdl29.net', 'mic.fr', 'migre.me', 'minu.me', 'moourl.com', 'more.sh', 'mut.lu', 'myurl.in', 'net.ms', 'net46.net', 'nicou.ch', 'nig.gr', 'notlong.com', 'nov.io', 'nq.st', 'nutshellurl.com', 'nxy.in', 'o-x.fr', 'okok.fr', 'onl.li', 'ou.af', 'ou.gd', 'oua.be', 'ouo.io', 'ow.ly', 'p.pw', 'parky.tv', 'past.is', 'pdh.co', 'ph.ly', 'pich.in', 'pin.st', 'plots.fr', 'plots.fr', 'pm.wu.cz', 'po.st', 'ppfr.it', 'ppst.me', 'ppt.cc', 'ppt.li', 'pqn.bz', 'prejit.cz', 'ptab.it', 'ptm.ro', 'pw2.ro', 'py6.ru', 'q.gs', 'qbn.ru', 'qqc.co', 'qr.net', 'qrtag.fr', 'qxp.cz', 'qxp.sk', 'r.cont.us', 'rb6.co', 'rcknr.io', 'rdz.me', 'redir.ec', 'redir.fr', 'redu.it', 'ref.so', 'reise.lc', 'relink.fr', 'repla.cr', 'ri.ms', 'riz.cz', 'rod.gs', 'roflc.at', 'rt.se', 's-url.fr', 'safe.mn', 'sagyap.tk', 'sdu.sk', 'seeme.at', 'segue.se', 'sh.st', 'sh.st', 'shar.as', 'shrinkurl.us', 'shorl.com', 'short.cc', 'short.ie', 'short.pk', 'shorte.st', 'shrt.in', 'shy.si', 'smu.sh', 'sicax.net', 'sina.lt', 'sk.gy', 'skr.sk', 'skroc.pl', 'smll.co', 'sn.im', 'snipurl.com', 'snsw.us', 'snurl.com', 'soo.gd', 'sort3.me', 'spn.sr', 'sq6.ru', 'ssl.gs', 'su.pr', 'surl.me', 'sux.cz', 'sy.pe', 't.cn', 't.co', 'ta.gd', 'tabzi.com', 'tau.pe', 'tdjt.cz', 'thesa.us', 'tighturl.com', 'tin.li', 'tini.cc', 'tiny.cc', 'tiny.lt', 'tiny.ms', 'tiny.pl', 'tinyurl.com', 'tinyurl.hu', 'tixsu.com', 'tldr.sk', 'tllg.net', 'tnij.org', 'tny.cz', 'to.ly', 'tohle.de', 'tpmr.com', 'tr.im', 'tr5.in', 'trck.me', 'trick.ly', 'trkr.ws', 'trunc.it', 'twet.fr', 'twi.im', 'twlr.me', 'twurl.nl', 'u.to', 'uby.es', 'ucam.me', 'ug.cz', 'ulmt.in', 'unlc.us', 'upzat.com', 'ur1.ca', 'url2.fr', 'url5.org', 'url.ie', 'url.likedeck.com', 'url.lotpatrol.com', 'urlcut.com', 'urlin.it', 'urls.fr', 'urltea.com', 'urlz.fr', 'urub.us', 'utfg.sk', 'v.gd', 'v.ht', 'v5.gd', 'vaaa.fr', 'valv.im', 'vaza.me', 'vbly.us', 'vd55.com', 'verd.in', 'vgn.me', 'vov.li', 'vsll.eu', 'vt802.us', 'vur.me', 'vv.vg', 'w1p.fr', 'waa.ai', 'wapurl.co.uk', 'wb1.eu', 'web99.eu', 'wed.li', 'wideo.fr', 'wp.me', 'wtc.la', 'wu.cz', 'ww7.fr', 'wwy.me', 'x.co', 'x.nu', 'x10.mx', 'x2c.eu', 'x2c.eumx', 'xav.cc', 'xgd.in', 'xib.me', 'xl8.eu', 'xoe.cz', 'xrl.us', 'xt3.me', 'xua.me', 'xub.me', 'xurls.co', 'yagoa.fr', 'yagoa.me', 'yatuc.com', 'yau.sh', 'yeca.eu', 'yect.com', 'yep.it', 'yogh.me', 'yon.ir', 'youfap.me', 'youtu.be', 'ysear.ch', 'yyv.co', 'z9.fr', 'zSMS.net', 'zapit.nu', 'zeek.ir', 'zip.net', 'zkr.cz', 'zkrat.me', 'zkrt.cz', 'zoodl.com', 'zpag.es', 'zti.me', 'zxq.net', 'zyva.org', 'zzb.bz']
    for url_string in file:
        url_string = url_string.strip()
        url_www = "www." + url_string
        match = url_token['host'].lower()
    
        if url_string == match or url_www == match:
            return 1
    return 0

def entropy(domain):
    frequency, L = Counter(domain), float(len(domain))
    return -sum( c/L * math.log(c/L, 2) for c in frequency.values())

def is_ip(url_token):
    try:
        if ipaddress.ip_address(url_token['host']):
            return 1
    except:
        return 0

character_set = ['.','-','_','/','?','=','@','&','!',' ','~',',','+','*','#','$','%', '//']
len_url_label_set = len(character_set)

url_label_character   = ['url_dot','url_hyphen','url_underscore','url_slash','url_question_mark','url_equals_to','url_at_rate','url_ampersand','url_exclamation','url_space','url_tilde','url_comma','url_plus','url_asterik','url_hash','url_dollar','url_percent', 'url_double_slash']
host_label_character  = ['host_dot','host_hyphen','host_underscore','host_slash','host_question_mark','host_equals_to','host_at_rate','host_ampersand','host_exclamation','host_space','host_tilde','host_comma','host_plus','host_asterik','host_hash','host_dollar','host_percent', 'host_double_slash']
path_label_character  = ['path_dot','path_hyphen','path_underscore','path_slash','path_question_mark','path_equals_to','path_at_rate','path_ampersand','path_exclamation','path_space','path_tilde','path_comma','path_plus','path_asterik','path_hash','path_dollar','path_percent', 'path_double_slash']
query_label_character = ['query_dot','query_hyphen','query_underscore','query_slash','query_question_mark','query_equals_to','query_at_rate','query_ampersand','query_exclamation','query_space','query_tilde','query_comma','query_plus','query_asterik','query_hash','query_dollar','query_percent', 'query_double_slash']
file_label_character =  ['file_dot','file_hyphen','file_underscore','file_slash','file_question_mark','file_equals_to','file_at_rate','file_ampersand','file_exclamation','file_space','file_tilde','file_comma','file_plus','file_asterik','file_hash','file_dollar','file_percent', 'file_double_slash']
url_label_ext1 = ['length_url','length_host','length_path','length_query','length_file', 'presence_of_path']
url_label_ext2  = ['presence_of_email','number of digits','number_of_tokens','number_of_tld','extension','presence_of_tld','length_of_parameters', 'presence_of_query']
len_url_label_ext1 = len(url_label_ext1)
len_url_label_ext2 = len(url_label_ext2)
host_label_ext = ['url_shortened','server_client','number_of_vowel', 'domain_entropy', 'is_host_ip']
len_host_label_ext = len(host_label_ext)
label = url_label_character + url_label_ext1 + url_label_ext2 + host_label_character + host_label_ext + path_label_character + query_label_character + file_label_character
# print(*label, sep = "', '")

def url_feature_generation(url_string):
    url_label_character_values    = [0 for i in range(len_url_label_set)]
    host_label_character_values   = [0 for i in range(len_url_label_set)]
    path_label_character_values   = [0 for i in range(len_url_label_set)]
    query_label_character_values  = [0 for i in range(len_url_label_set)]
    file_label_character_values   = [0 for i in range(len_url_label_set)]
    url_label_ext1_val = [0 for i in range(len_url_label_ext1)]
    url_label_ext2_val = [0 for i in range(len_url_label_ext2)]
    host_label_ext_val = [0 for i in range(len_host_label_ext)]
    
    url_token = split_url(url_string)
    
    url_label_ext1_val[0] = len(url_string)
    url_label_ext1_val[1] = len(url_token['host'])
    url_label_ext1_val[2] = len(url_token['path'])
    url_label_ext1_val[3] = len(url_token['query'])
    url_label_ext1_val[4] = len(posixpath.basename(url_token['path']))
    url_label_ext1_val[5] = int(len(url_token['path']) > 1) # 1 due to '/' at the end
    
    url_label_ext2_val[0] = email_in_text(url_string)
    url_label_ext2_val[1] = sum(c.isdigit() for c in url_string)
    url_label_ext2_val[2] = len(url_tokenizer(url_string))
    url_label_ext2_val[3] = count_tld(url_string)
    url_label_ext2_val[4] = str(extract_extension(posixpath.basename(url_token['path']))) if url_token['path'] else '?'
    url_label_ext2_val[5] = check_tld(url_token['host'])
    url_label_ext2_val[6] = count_params(url_token['query'])
    url_label_ext2_val[7] = int(len(url_token['query']) > 1) # 1 due to '=' at the end
    
    host_label_ext_val[0] = check_url_shortener(url_token)
    host_label_ext_val[1] = check_server_client(url_string)
    host_label_ext_val[2] = count_vowels(url_token['host'])
    host_label_ext_val[3] = entropy(url_token['host'])
    host_label_ext_val[4] = is_ip(url_token)
 
    for i in range(len_url_label_set):
        url_label_character_values[i]    = count(url_token['url'], character_set[i])
        host_label_character_values[i]   = count(url_token['host'], character_set[i])
        path_label_character_values[i]   = count(url_token['host'], character_set[i])
        file_label_character_values[i]   = count(posixpath.basename(url_token['host']), character_set[i])
        query_label_character_values[i]  = count(url_token['query'], character_set[i])
        
    url_feature_set = url_label_character_values + url_label_ext1_val + url_label_ext2_val
    host_feature_set = host_label_character_values + host_label_ext_val 
    path_query_file_feature_set = path_label_character_values + query_label_character_values + file_label_character_values
    
    feature_set = url_feature_set + host_feature_set + path_query_file_feature_set
    
    return feature_set


# In[3]:


def ext_one_hot_encoding(extension):
    extension_list = ['apk', 'asp', 'aspx', 'avi', 'bat', 'bin', 'c', 'cab', 'cfm', 'cgi', 'class', 'com', 'cpl', 'cpp', 'css', 'dat', 'dds', 'dll', 'doc', 'docx', 'exe', 'gif', 'gz', 'h', 'htm', 'html', 'ico', 'jar', 'jpg', 'js', 'jsp', 'lua', 'm', 'mov', 'mp3', 'mp4', 'mpg', 'pdf', 'php', 'pl', 'png', 'py', 'rar', 'rss', 'sh', 'svg', 'swf', 'sys', 'tar', 'tmp', 'torrent', 'txt', 'xhtml', 'xls', 'xml', 'zip']
    final_list = [0 for ext in extension_list]
    try:
        final_list[extension_list.index(extension)] = 1
    except:
        pass
    return final_list


# In[4]:


initial_labels = ['url_dot', 'url_hyphen', 'url_underscore', 'url_slash', 'url_question_mark', 'url_equals_to', 'url_at_rate', 'url_ampersand', 'url_exclamation', 'url_space', 'url_tilde', 'url_comma', 'url_plus', 'url_asterik', 'url_hash', 'url_dollar', 'url_percent', 'url_double_slash', 'length_url', 'length_host', 'length_path', 'length_query', 'length_file', 'presence_of_path', 'presence_of_email', 'number of digits', 'number_of_tokens', 'number_of_tld', 'extension', 'presence_of_tld', 'length_of_parameters', 'presence_of_query', 'host_dot', 'host_hyphen', 'host_underscore', 'host_slash', 'host_question_mark', 'host_equals_to', 'host_at_rate', 'host_ampersand', 'host_exclamation', 'host_space', 'host_tilde', 'host_comma', 'host_plus', 'host_asterik', 'host_hash', 'host_dollar', 'host_percent', 'host_double_slash', 'url_shortened', 'server_client', 'number_of_vowel', 'domain_entropy', 'is_host_ip', 'path_dot', 'path_hyphen', 'path_underscore', 'path_slash', 'path_question_mark', 'path_equals_to', 'path_at_rate', 'path_ampersand', 'path_exclamation', 'path_space', 'path_tilde', 'path_comma', 'path_plus', 'path_asterik', 'path_hash', 'path_dollar', 'path_percent', 'path_double_slash', 'query_dot', 'query_hyphen', 'query_underscore', 'query_slash', 'query_question_mark', 'query_equals_to', 'query_at_rate', 'query_ampersand', 'query_exclamation', 'query_space', 'query_tilde', 'query_comma', 'query_plus', 'query_asterik', 'query_hash', 'query_dollar', 'query_percent', 'query_double_slash', 'file_dot', 'file_hyphen', 'file_underscore', 'file_slash', 'file_question_mark', 'file_equals_to', 'file_at_rate', 'file_ampersand', 'file_exclamation', 'file_space', 'file_tilde', 'file_comma', 'file_plus', 'file_asterik', 'file_hash', 'file_dollar', 'file_percent', 'file_double_slash']
final_labels = ['url_dot', 'url_hyphen', 'url_underscore', 'url_slash', 'url_question_mark', 'url_equals_to', 'url_at_rate', 'url_ampersand', 'url_exclamation', 'url_space', 'url_tilde', 'url_comma', 'url_plus', 'url_asterik', 'url_hash', 'url_dollar', 'url_percent', 'url_double_slash', 'length_url', 'length_host', 'length_path', 'length_query', 'length_file', 'presence_of_path', 'presence_of_email', 'number of digits', 'number_of_tokens', 'number_of_tld', 'presence_of_tld', 'length_of_parameters', 'presence_of_query', 'host_dot', 'host_hyphen', 'host_underscore', 'host_slash', 'host_question_mark', 'host_equals_to', 'host_at_rate', 'host_ampersand', 'host_exclamation', 'host_space', 'host_tilde', 'host_comma', 'host_plus', 'host_asterik', 'host_hash', 'host_dollar', 'host_percent', 'host_double_slash', 'url_shortened', 'server_client', 'number_of_vowel', 'domain_entropy', 'is_host_ip', 'path_dot', 'path_hyphen', 'path_underscore', 'path_slash', 'path_question_mark', 'path_equals_to', 'path_at_rate', 'path_ampersand', 'path_exclamation', 'path_space', 'path_tilde', 'path_comma', 'path_plus', 'path_asterik', 'path_hash', 'path_dollar', 'path_percent', 'path_double_slash', 'query_dot', 'query_hyphen', 'query_underscore', 'query_slash', 'query_question_mark', 'query_equals_to', 'query_at_rate', 'query_ampersand', 'query_exclamation', 'query_space', 'query_tilde', 'query_comma', 'query_plus', 'query_asterik', 'query_hash', 'query_dollar', 'query_percent', 'query_double_slash', 'file_dot', 'file_hyphen', 'file_underscore', 'file_slash', 'file_question_mark', 'file_equals_to', 'file_at_rate', 'file_ampersand', 'file_exclamation', 'file_space', 'file_tilde', 'file_comma', 'file_plus', 'file_asterik', 'file_hash', 'file_dollar', 'file_percent', 'file_double_slash', 'extension -- apk', 'extension -- asp', 'extension -- aspx', 'extension -- avi', 'extension -- bat', 'extension -- bin', 'extension -- c', 'extension -- cab', 'extension -- cfm', 'extension -- cgi', 'extension -- class', 'extension -- com', 'extension -- cpl', 'extension -- cpp', 'extension -- css', 'extension -- dat', 'extension -- dds', 'extension -- dll', 'extension -- doc', 'extension -- docx', 'extension -- exe', 'extension -- gif', 'extension -- gz', 'extension -- h', 'extension -- htm', 'extension -- html', 'extension -- ico', 'extension -- jar', 'extension -- jpg', 'extension -- js', 'extension -- jsp', 'extension -- lua', 'extension -- m', 'extension -- mov', 'extension -- mp3', 'extension -- mp4', 'extension -- mpg', 'extension -- pdf', 'extension -- php', 'extension -- pl', 'extension -- png', 'extension -- py', 'extension -- rar', 'extension -- rss', 'extension -- sh', 'extension -- svg', 'extension -- swf', 'extension -- sys', 'extension -- tar', 'extension -- tmp', 'extension -- torrent', 'extension -- txt', 'extension -- xhtml', 'extension -- xls', 'extension -- xml', 'extension -- zip']


# In[5]:


url_string = 'https://drive.google.com/drive/folders/1a-sm43p4W9J4mWo8gksOCZnFFieK0zNy' #input


# In[6]:


feature_set = url_feature_generation(url_string)


# In[7]:


index_of_ext = initial_labels.index('extension')


# In[8]:


ext = feature_set[index_of_ext]


# In[9]:


feature_set.pop(index_of_ext)


# In[10]:


ext_one_hot_list = ext_one_hot_encoding(ext)


# In[11]:


feature_set += ext_one_hot_list


# In[12]:


features_df = pd.DataFrame([feature_set], columns = final_labels)
print(features_df)


# In[13]:


time.time()-start_time

