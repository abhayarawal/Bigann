(defproject bigann "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0-beta2"]
                [org.clojure/core.async "0.1.346.0-17112a-alpha"]
  							[criterium "0.4.3"]
                [net.mikera/vectorz-clj "0.36.0"]
								[net.mikera/core.matrix "0.42.1"]]
  :main ^:skip-aot bigann.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}}
  :jvm-opts ["-Xmx1g" "-server"])
