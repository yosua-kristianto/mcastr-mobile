from dotenv import load_dotenv

def main():
    load_dotenv()

    from model.mobile_bert import MobileBERT
    MobileBERT().train()

    return 0

if __name__ == "__main__":
    from datetime import datetime

    print(f"""
                                                          
   *                             
 (  `            )     (         
 )\))(        ( /( (   )\   (    
((_)()\   (   )\()))\ ((_) ))\   
(_()((_)  )\ ((_)\((_) _  /((_)  
|  \/  | ((_)| |(_)(_)| |(_))    
| |\/| |/ _ \| '_ \| || |/ -_)   
|_|  |_|\___/|_.__/|_||_|\___|   
               (                 
      (        )\ )  *   )       
    ( )\  (   (()/(` )  /(       
    )((_) )\   /(_))( )(_))      
   ((_)_ ((_) (_)) (_(_())       
    | _ )| __|| _ \|_   _|       
    | _ \| _| |   /  | |         
    |___/|___||_|_\  |_|         
                                  
    Session {datetime.now()}
    """)

    main()