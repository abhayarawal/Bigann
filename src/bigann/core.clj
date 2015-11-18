(ns bigann.core
  (:gen-class)
  (:require [clojure.core.async
							:as async
							:refer [>! <! >!! <!! go chan buffer close! thread
							       alts! alts!! timeout]]))

(use 'criterium.core)
(use 'clojure.core.matrix)
(require '[clojure.core.matrix.operators :as Mat]
				 '[clojure.java.io :as io]
				 '[clojure.string :as str]
				 '[clojure.core.reducers :as r])

(set-current-implementation :vectorz)
; (set! *warn-on-reflection* true)

(def loc "./dumps/")

(def file-list
	"Creates a list of all dump files"
	(->>
		(file-seq (clojure.java.io/file loc))
		(filter #(.isFile %))
		(pmap #(.getName %))
		(filter #(.endsWith ^String % ".txt"))))

(defn min-max
	[func & seq-arg]
		(loop [acc (transient []) t seq-arg]
			(if (empty? (first t))
				(rseq (persistent! acc))
				(recur (conj! acc (apply func (map peek t))) (map pop t)))))

(def min-fold 
	(partial min-max min))

(def max-fold 
	(partial min-max max))

(defn gen-matrix
	"Generates a random matrix based on inputs x, y"
	[^ints r ^ints c]
	(for [_ (take r (range))] (for [_ (take c (range))] (* (if (< 0.5 (rand)) -1 1) (rand)))))

(defn fetch-segment
	[]
	(let [ch (chan)]
		(go
			(loop [[h & t] (shuffle file-list)]
				(when-not (nil? h)
					(>! ch
						(with-open [^java.io.BufferedReader reader (io/reader (str loc h))]
							(doall (line-seq reader))))
					(recur t)))
			(close! ch))
		ch))

(defn parser [^String line]
	(let [[x1 x2 _ _ x5 _ x7 x8 x9 x10 x11 x12 x13 x14 x15] 
					(-> 
						line
						(str/lower-case)
						(str/replace #"nan" "0")
						(str/split #"[,]+")
						(drop-last))]

		(map (fn [^String item] 
						(if (= 0.0 (Float. item))
							0.0001
							(Float. item)))
			[x7 x9 x10 x12 x14 x13])))

(defn async-fetch
	[]
	(let [ch (chan)]
		(go
			(let [ch2 (fetch-segment)]
				(loop []
					(when-let [lines (<!! ch2)]
						(let [rows
										(->>
											lines
											(map parser)
											(filter #(not= (last %) 0.0001))
											(shuffle))]
							(doseq [line rows]
								(>! ch line)))
						(recur))))
			(close! ch))
		ch))

(defmacro read-data
	"Macro for accessing data"
	[ch & exprs]
	`(let [~ch (async-fetch)]
			~@exprs))

(defmacro forward
	"Macro for eliminating repeating code when reading data"
	[line & exprs]
	`(read-data ch#
		(loop []
			(when-let [~line (<!! ch#)]
				~@exprs
				(recur)))))

(defn fetch-normalizer []
	"Returns a normalizer function"
	(read-data ch
		(loop [max-acc (<!! ch) min-acc max-acc]
			(if-let [line (<!! ch)]
				(recur 
					(max-fold (vec max-acc) (vec line))
					(min-fold (vec min-acc) (vec line)))
				(fn [X]
					(Mat//
						(Mat/- X min-acc)
						(Mat/- max-acc min-acc)))))))

(def td-normals (future (fetch-normalizer)))

(def i-layer-sz "Input layer size" 5)
(def h-layer-sz "Hidden 1 layer size" 120)
(def o-layer-sz "Output layer size" 1)
(def iteration "Total iteration for training" 250)
(def rate "Rate of change of weights" 0.9)

(def w1 (future (gen-matrix i-layer-sz h-layer-sz)))
(def w2 (future (gen-matrix h-layer-sz o-layer-sz)))

(defn expn
	"Computes the exponential of a matrix"
	[z]
	(map (fn [l] (map #(java.lang.Math/exp %) l)) z))

(defn -ve
	"Negates a given matrix"
	[z]
  (Mat/* -1 z))

(defn ** 
	"Powers a base based on arguments (x ^ y)"
	[base ex] (java.lang.Math/pow base ex))

(defn sq 
	"Squares a number"
	[n] (** n 2))

(defn map-mat
	"Runs a given function on all elements of a matrix"
	[func z]
  (map (fn [l] (map #(func %) l)) z))

(defn sigmoid 
	"Runs sigmoid activation function on a matrix"
	[z]
	(Mat// 1 (Mat/+ 1 (expn (-ve z)))))

(defn sigmoid-prime
	"Runs derivative of sigmoid function on a matrix"
	[z]
	(Mat// (expn (-ve z)) (map-mat sq (Mat/+ 1 (expn (-ve z))))))

(defn feed
	"Feed forwards a row with given weight and returns yHat"
	[row w1 w2]
	(let [x [(pop row)]
				y [[(peek row)]]
				z2 (dot x w1)
				a2 (sigmoid z2)
				z3 (dot a2 w2)
				yHat (sigmoid z3)]
		yHat))

(defn feed-forward
	[row w1 w2]
	(let [x [(pop row)]
				y [[(peek row)]]
				z2 (dot x w1)
				a2 (sigmoid z2)
				z3 (dot a2 w2)
				yHat (sigmoid z3)]

		(let [delta3 (Mat/* (-ve (Mat/- y yHat)) (sigmoid-prime z3))
					djdw2 (dot (transpose a2) delta3)
					delta2 (Mat/* (dot delta3 (transpose w2)) (sigmoid-prime z2))
					djdw1 (dot (transpose x) delta2)
					deltaW2 (Mat/- w2 (Mat/* rate djdw2))
					deltaW1 (Mat/- w1 (Mat/* rate djdw1))
					[[ error ]] (Mat/- y yHat)]

			(let [[[ error-2 ]] (Mat/- y (feed row deltaW1 deltaW2))]
				(cond
					(< error 0)
						(if (< error-2 error)
							{:w1 w1 :w2 w2} 
							{:w1 deltaW1 :w2 deltaW2})
					:else
						(if (< error-2 error) 
							{:w1 deltaW1 :w2 deltaW2} 
							{:w1 w1 :w2 w2}))))))

; (defn propagate
; 	[wh1 wh2]
; 	)

(defn propagate
	[wh1 wh2]
	(read-data ch
		(loop [acc wh1 acc2 wh2]
			(if-let [line (<!! ch)]
				(let [ret (feed-forward (vec (@td-normals line)) acc acc2)]
					(recur (:w1 ret) (:w2 ret)))
				{:w1 acc :w2 acc2}))))

(def weight
	(loop [n 0 acc @w1 acc2 @w2]
		(if (< n iteration)
			(let [ret (propagate acc acc2)]
				(recur (inc n) (:w1 ret) (:w2 ret)))
			{:w1 acc :w2 acc2})))

(forward line
	(let [normalized (vec (@td-normals line))
				yHat (feed normalized (:w1 weight) (:w2 weight))]
		(pm yHat)
		(pm (last normalized))))

(defn -main
  [& args]
  (println "\n Exiting"))


; (0.62849 0.48801985 512.1824 0.194 9.0 0.845 -3.237 1.0 0.514 79.709 4.0 1.0E-4 0.5394108)
