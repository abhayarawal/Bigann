(defn dump-process-fn
	[^String line]
	(let [new-line (str/replace line #"[^\w,\.]+" "")]
		(cond
			(.contains new-line "F")
				(-> 
					new-line
					(str/replace #"F," "")
					(str ",F" "\n"))
			(.contains new-line "M")
				(-> 
					new-line
					(str/replace #"M," "")
					(str ",M" "\n")))))

(defn create-dump
	[in]
	(let [out (gensym "dump")]
		(with-open [^java.io.BufferedReader rdr (io/reader in)
								^java.io.BufferedWriter wtr (io/writer (str "./src/ann/" out ".txt"))]
			(let [lines (line-seq rdr)]
				(dorun
					(map (fn [^String line] (.write wtr line))
						(pmap dump-process-fn lines)))))
		out))

(defn load-dump []
	(with-open [^java.io.BufferedReader rdr (io/reader "./dump/dump.txt")]
		(let [lines (doall (shuffle (line-seq rdr)))]
			lines)))

(defn write-dump []
	(let [parts (partition 50 (load-dump))]
		(loop [[h & t] parts]
			(if-not (nil? h)
				(do
					(with-open [^java.io.BufferedWriter wtr (io/writer (str "./dumps/" (gensym "dump") ".txt"))]
						(dorun
							(map (fn [^String line] (do (.write wtr line) (.newLine wtr))) (shuffle h))))
					(recur t))))))