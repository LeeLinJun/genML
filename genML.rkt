#lang racket
; dictionary of symbols
(define ht (make-hash))

(define (nick namepair)
  (hash-set! ht (first namepair) (second namepair)))
;; often used imports
(define scriptInit "")
(nick '("plot" "import matplotlib.pyplot as plot"))
(nick '("mnist" "from keras.datasets import mnist "))
(nick '("kut" "from keras.utils import np_utils"))
(nick '("np" "import numpy as np~nnp.random.seed(1337)"))
(nick '("model" "from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout"))
(nick '("conv" "from keras.layers import Conv2D,MaxPooling2D,Flatten") )
;;optimizers
(nick '("rmsprop" "from keras.optimizers import RMSprop"))
(nick '("adam" "from keras.optimizers import Adam") )

; imports
(define (add2script  contents script)
  (if (null? contents)
      script
      (add2script
       (cdr contents)
       (string-append
                    script
                    (hash-ref ht (car contents))
                    "~n"))))


;;preprocess

(define (mnist-init [XTrReshape "X_train.reshape(X_train.shape[0],-1)/255 "] [XTeReshape "X_test.reshape(X_test.shape[0],-1)/255"])
  (string-append
  "(X_train,y_train),(X_test,y_test) = mnist.load_data()
X_train = " XTrReshape "~nX_test = " XTeReshape "~ny_train=np_utils.to_categorical(y_train,num_classes=10)
y_test=np_utils.to_categorical(y_test,num_classes=10)
")
  )

;build
(define (fc act dim [type "Dense"] [input 0])
  (string-append
   type "(" (number->string dim)
   (if (= input 0)
       ""
       (string-append ",input_dim="  (number->string input)))
   ",activation='" act "'),~n"))

(define (c2 filters dimx dimy act [inputx 0] [inputy 0] [channel 0] [data-format ""] [padding ""])
  (string-append
   "Conv2D(" (number->string filters) ",(" (number->string dimx) "," (number->string dimy) "),activation='" act "'"
   (if (= inputx 0)
       ""
       (if (= 0 (string-length data-format))
                          (string-append ",input_shape=(" (number->string inputx) "," (number->string inputy) "," (number->string channel) "),")
                          (string-append ",input_shape=(" (number->string channel) ","(number->string inputx) "," (number->string inputy) ","  ")," data-format)
                          ))
       
                      
    (if (> (string-length padding) 0) ",padding='" "") padding (if (> (string-length padding) 0) "'" "") "),\n"))


(define (mp2 px py #:strides [strides ""] padding )
  (string-append
   "MaxPooling2D(pool_size=(" (number->string px) "," (number->string py) ")," strides "padding='" padding "'),\n" ))

(define (d p)
  (string-append "Dropout(" (number->string p) "),\n" ))
(define f "Flatten(),\n")


(define (build-model layers loss #:optparameter [optparameter "(lr=1e-4)"]  #:opt [opt "sgd"] )
  (string-append
   
   "model = Sequential([~n"
   layers
   "])~n"
   "opt=" opt optparameter
   "~nmodel.compile(
    optimizer = opt,
    loss='" loss "',
    metrics=['accuracy'],
)"
   ))

;train
(define (gen-train epoch batch_size)
  (string-append
   "~nprint('Training ----')
model.fit(X_train,y_train,epochs=" (number->string epoch) ",batch_size=" (number->string batch_size)")"))
;test
(define (gen-test)
  (string-append
   "~nprint('Testing ----')
loss,accuracy= model.evaluate(X_test,y_test,verbose=0)
print('test loss',loss)
print('test accuracy',accuracy)"))
; write-file
(define out (open-output-file "./ML.py" #:exists 'replace))

(define imports
   '("mnist"  "np" "kut" "model" "adam" "conv"
;preprocess
              ))


  
(define layers
    (string-append
     (c2 32 3 3 "relu" 28 28 1 "data_format='channels_first'" "same")
     (c2 64 3 3 "relu" 28 28 1 "data_format='channels_first'" "same")
     (mp2 2 2 #:strides "strides=(2,2)," "same")
     (d 0.25)
     f
     (fc "relu" 128)
     (d 0.5)
     (fc "softmax" 10)
     ;(fc  "relu" 32 784)
     ;(fc  "softmax" 10)
     ))


  
(fprintf out
         (string-append
         (add2script  imports scriptInit)
         (mnist-init "X_train.reshape(-1,1,28,28)" "(X_test.reshape(-1,1,28,28))")
         (build-model layers "categorical_crossentropy" #:opt "Adam"
                      ;#:optparameter "(lr=0.001,rho=0.9,epsilon=1e-08,decay=0.0)")
                      )
         (gen-train 5 32)
         (gen-test)))
(close-output-port out)
