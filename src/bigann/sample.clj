(ns ann.core
  (:gen-class)
  (:require [clojure.core.async
							:as async
							:refer [>! <! >!! <!! go chan buffer close! thread
							       alts! alts!! timeout]]))

(use 'criterium.core)
(use 'clojure.core.matrix)
(require '[clojure.core.matrix.operators :as Mat]
				 '[clojure.java.io :as io]
				 '[clojure.string :as str])

(set-current-implementation :vectorz)
; (set! *warn-on-reflection* true)

;; needs exception handling
(def loc "./dumps/")

(def file-list
	"Creates a list of all dump files"
	(->>
		(file-seq (clojure.java.io/file loc))
		(filter #(.isFile %))
		(pmap #(.getName %))
		(filter #(.endsWith ^String % ".txt"))))

(def data-map 
	"Fix for nominal data"
	{:b 0.09 
	 :o 0.01
	 :m 1.0
	 :f 0.0})

(defn max-fold
	"Finds the maximum relative elements on given vectors"
	[& seq-arg]
	(loop [acc (transient []) t seq-arg]
		(if (empty? (first t))
			(rseq (persistent! acc))
			(recur (conj! acc (apply max (map peek t))) (map pop t)))))

(defn gen-matrix
	"Generates a random matrix based on inputs x, y"
	[^ints r ^ints c]
	(for [_ (take r (range))] (for [_ (take c (range))] (/ (rand 10) 10))))

(defn async-fetch-set
	"Creates and returns a channel lazily accessed random row. Channel is closed once all dump files are read. List if shuffled before reading"
	[]
	(let [ch (chan)]
		(go
			(loop [[h & t] (shuffle file-list)]
				(if-not (nil? h)
					(do 
						(with-open [^java.io.BufferedReader reader (io/reader (str loc h))]
							(doseq [^String line (line-seq reader)]
								(>! ch line)))
						(recur t))))
			(close! ch))
		ch))

(defn parser
	"Parses a line from the file and returns a list"
	[^String line]
	(let ;; id  sp i  fl rw cl cw bd sex
		[[x1 _ _ & t]
		(->>
			(->
				line
				(str/lower-case)
				(str/replace #"[^\w,\.]+" "")
				(str/split #"[\s,]+"))

			(map (fn [^String item]
				(or ((keyword item) data-map) (Float. item)))))]
		(conj t x1)))

(defmacro read-data
	"Macro for accessing data"
	[ch & exprs]
	`(let [~ch (async-fetch-set)]
			~@exprs))

(defmacro forward
	"Macro for eliminating repeating code when reading data"
	[row & exprs]
	`(read-data ch#
		(loop []
			(when-let [line# (<!! ch#)]
				(let [[_# & ~row] (parser line#)]
					~@exprs)
				(recur)))))

(defn fetch-normalizer []
	"Reads all files and create a new vector with maximum elements for normalizer"
	(read-data ch
		(loop [acc (parser (<!! ch))]
			(if-let [line (<!! ch)]
				(recur (max-fold (vec acc) (vec (parser line))))
				(let [[_ & t] acc]
					(map #(/ 1.0 %) t))))))

; math
(def i-layer-sz "Input layer size" 5)
(def o-layer-sz "Output layer size" (do 1))
(def iteration "Total iteration for training" (do 180))
(def rate "Rate of change of weights" (do 0.00275))
(def threshold "Threshold for binary result" (do 0.4941))

(def td-normals "Fetches normalizer in a separate thread" (future (fetch-normalizer)))
; (def w1 (future (gen-matrix i-layer-sz o-layer-sz)))
(def w1 "Literal store of initial weights" 
	(future (do [[0.4339311459465581] [-1.2696008833899244] [0.7180165303171447] [0.10714482114075632] [-0.07062929913609142]])))

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

;; Start
(defn feed
	"Feed forwards a row with given weight and returns yHat"
	[row w]
	(let [x [(pop row)]
				y [[(peek row)]]
				z1 (dot x w)
				yHat (sigmoid z1)]
		yHat))

(defn feed-forward
	"Feed forwards a row with given weight and returns new weight"
	([row w]
		(let [x [(pop row)]
					y [[(peek row)]]
					z1 (dot x w)
					yHat (sigmoid z1)
					[[error]] (Mat/- y yHat)
					[[delta]] (dot (-ve (Mat/- y yHat)) (sigmoid-prime z1))
					djdw (Mat/* delta (transpose x))
					w+1 (Mat/- w (Mat/* rate djdw))]

			(let [yHat+1 (feed row w+1)
						[[error-2]] (Mat/- y yHat+1)]
				(cond
					(< error 0)
						(if (< error-2 error) w w+1)
					:else
						(if (< error-2 error) w+1 w))))))

(defn propagate
	"Runs feed forward on all rows of the dataset"
	([wh]
		(read-data ch
			(loop [acc wh]
				(if-let [line (<!! ch)]
					(let [[_ & row] (parser line)
								wl (feed-forward (->> @td-normals (Mat/* row)) acc)]
						(recur wl))
					acc)))))

(println "\n"
				 "Training initialized w/" "\n" 
				 "Rate:" rate "\n"
				 "Iterations:" iteration "\n"
				 "Threshold:" threshold "\n"
				 "Initial weights:" @w1 "\n"
				 "Input size:" i-layer-sz "\n"
				 "Output size:" o-layer-sz "\n")

(def weight
	"Runs propagation on a loop and stores the final weights"
	(loop [n 0 acc @w1]
		(if (< n iteration)
			(recur (inc n) (propagate acc))
			acc)))

(println " Final weights:" weight "\n")

; ; ; Test
(forward row
	(let [[[yHat]] (feed (->> @td-normals (Mat/* row)) weight)
				y (last row)
				error? (= (if (> y threshold) 0 1)
							 (if (> yHat threshold) 0 1))]
		(println "" row 
						 " | yHat:" (format "%.2f" yHat) 
						 " | Actual sex:" (if (= y 1.0) "Male" "Female")
						 " | Predicted sex:" (if (> yHat threshold) "Male" "Female"))
		(if-not error? (println " ERROR: ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"))))

(defn -main
  [& args]
  (println "\n Exiting"))
  ; (System/exit 0))





