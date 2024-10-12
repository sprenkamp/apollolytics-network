import asyncio
from twscrape import Account, AccountsPool, API, gather
from twscrape.logger import set_log_level
import yaml

async def main(accounts):
    pool = AccountsPool()  # or AccountsPool("path-to.db") - default is `accounts.db` 

    for account in accounts:
        await pool.add_account(
            account["x-user-name"], 
            account["x-password"], 
            account["email-address"], 
            account["email-password"]
        )

    # log in to all new accounts
    await pool.login_all()

    api = API(pool)

    results = await gather(api.search("ukrainian war", limit=20)) #limit now working as expected, doesn't act as max_results but used as check of "have enough results being returned yet?" AKA => if first batch is bigger than limit, the full first batch is returned
    print(len(results))
    for tweet in results:
        print(tweet.id, tweet.user.username, tweet.rawContent)
        print()

    # search api (latest tab)
    # await gather(api.search("ukrainian war", limit=20))  # list[Tweet]

    # # graphql api
    # tweet_id, user_id, user_login = 20, 2244994945, "twitterdev"

    # await api.tweet_details(tweet_id)  # Tweet
    # await gather(api.retweeters(tweet_id, limit=20))  # list[User]
    # await gather(api.favoriters(tweet_id, limit=20))  # list[User]

    # await api.user_by_id(user_id)  # User
    # await api.user_by_login(user_login)  # User
    # await gather(api.followers(user_id, limit=20))  # list[User]
    # await gather(api.following(user_id, limit=20))  # list[User]  
    # await gather(api.user_tweets(user_id, limit=20))  # list[Tweet]
    # await gather(api.user_tweets_and_replies(user_id, limit=20))  # list[Tweet]

    # note 1: limit is optional, default is -1 (no limit)
    # note 2: all methods have `raw` version e.g.:

    # async for tweet in api.search("ukrainian war", limit=25):
    #     print(tweet.id, tweet.user.username, tweet.rawContent)  # tweet is `Tweet` object
    #     print()

    # async for rep in api.search_raw("ukrainian war"):
    #     print(rep.status_code, rep.json())  # rep is `httpx.Response` object

    # # change log level, default info
    # set_log_level("DEBUG")

    # # Tweet & User model can be converted to regular dict or json, e.g.:
    # doc = await api.user_by_id(user_id)  # User
    # doc.dict()  # -> python dict
    # doc.json()  # -> json string

if __name__ == "__main__":

    stream = open('accounts.yml', 'r')
    accounts = yaml.load(stream, Loader=yaml.FullLoader)

    asyncio.run(main(accounts["users"]))